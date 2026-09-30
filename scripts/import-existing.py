#!/usr/bin/env python3
"""Import existing and curated company documents into the local knowledge base."""

import argparse
import asyncio
from pathlib import Path

from company_agent.config import PROJECT_ROOT, get_settings
from company_agent.database import KnowledgeRepository
from company_agent.document_parser import SUPPORTED_EXTENSIONS
from company_agent.ollama_client import OllamaClient
from company_agent.services import KnowledgeService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量导入本地知识库资料")
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "uploads", PROJECT_ROOT / "knowledge" / "curated"],
        help="一个或多个待导入目录，默认使用 uploads/ 和 knowledge/curated/",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    source_dirs = [path.expanduser().resolve() for path in args.directories]
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            print(f"目录不存在：{source_dir}")
            return 1

    settings = get_settings()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        model_context=settings.model_context,
    )
    service = KnowledgeService(settings, repository, client)

    files = sorted(
        (
            path
            for source_dir in source_dirs
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path),
    )
    if not files:
        print("指定目录中没有找到可导入文件")
        return 1

    succeeded = 0
    failed = 0
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}")
        try:
            result = await service.ingest_path(path)
        except Exception as exc:  # noqa: BLE001 - batch importer must continue
            failed += 1
            print(f"  失败：{exc}")
            continue
        succeeded += 1
        marker = "已跳过（内容重复）" if result["duplicate"] else "导入成功"
        document = result["document"]
        chunk_count = document.get("chunk_count", 0) if isinstance(document, dict) else 0
        media_count = document.get("media_count", 0) if isinstance(document, dict) else 0
        print(f"  {marker}，知识片段：{chunk_count}，图片：{media_count}")

    print(f"\n完成：成功/跳过 {succeeded}，失败 {failed}，总计 {len(files)}。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
