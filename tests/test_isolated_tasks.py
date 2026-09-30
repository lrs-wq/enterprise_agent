import asyncio
import os
from pathlib import Path

import pytest
from openpyxl import Workbook

from company_agent.config import Settings
from company_agent.database import KnowledgeRepository
from company_agent.isolated_tasks import (
    extract_document_media_isolated,
    parse_document_isolated,
)
from company_agent.services import KnowledgeService


async def test_isolated_parser_preserves_spreadsheet_locations(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "产品表"
    sheet.append(["产品名称", "卖点"])
    sheet.append(["宠物饮水机", "循环过滤"])
    source = tmp_path / "products.xlsx"
    workbook.save(source)

    chunks = await parse_document_isolated(source, max_spreadsheet_rows=10)

    assert len(chunks) == 1
    assert chunks[0].sheet == "产品表"
    assert chunks[0].row_start == 2
    assert "循环过滤" in chunks[0].content


async def test_isolated_parser_preserves_validation_errors(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["标题"])
    sheet.append(["第一行"])
    sheet.append(["第二行"])
    source = tmp_path / "too-many.xlsx"
    workbook.save(source)

    with pytest.raises(ValueError, match="限制"):
        await parse_document_isolated(source, max_spreadsheet_rows=1)


async def test_isolated_media_extraction_uses_final_paths(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "uploads" / "Ckin Pet宣传册.xls"
    if not source.exists():
        pytest.skip("本地 WPS 宣传册样本不存在")

    output_dir = tmp_path / "media"
    assets = await extract_document_media_isolated(source, output_dir)

    assert len(assets) == 19
    assert all(Path(asset.stored_path).parent == output_dir for asset in assets)
    assert all(Path(asset.stored_path).exists() for asset in assets)


async def test_isolated_task_cancellation_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from company_agent import isolated_tasks

    class FakeProcess:
        returncode: int | None = None
        pid = 12345
        terminated = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        async def wait(self) -> int:
            return -15

    process = FakeProcess()
    if isolated_tasks.os.name == "posix":

        def fake_killpg(_pid: int, _signal: object) -> None:
            process.terminated = True
            process.returncode = -15

        monkeypatch.setattr(isolated_tasks.os, "killpg", fake_killpg)

    async def fake_create(*_args: object, **_kwargs: object) -> FakeProcess:
        return process

    monkeypatch.setattr(isolated_tasks.asyncio, "create_subprocess_exec", fake_create)
    task = asyncio.create_task(parse_document_isolated(tmp_path / "sample.txt"))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated


async def test_blocked_real_parser_process_is_cancelled(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("需要 POSIX FIFO")
    source = tmp_path / "blocked.txt"
    os.mkfifo(source)
    task = asyncio.create_task(parse_document_isolated(source))
    await asyncio.sleep(0.5)
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)


async def test_document_ingest_uses_isolated_parser_end_to_end(tmp_path: Path) -> None:
    class FakeEmbeddings:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    source = tmp_path / "company.txt"
    source.write_text("公司主营宠物用品研发与销售。", encoding="utf-8")
    service = KnowledgeService(settings, repository, FakeEmbeddings())  # type: ignore[arg-type]

    outcome = await service.ingest_path(source)

    document = outcome["document"]
    assert isinstance(document, dict)
    assert document["status"] == "READY"
    assert document["publication_status"] == "draft"
    assert len(repository.list_chunks(str(document["id"]))) == 1
