"""Cancelable subprocesses for synchronous document parsing and media extraction."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, cast

from company_agent.document_parser import ParsedChunk, parse_document
from company_agent.media import ExtractedMedia, extract_document_media


def _worker(operation: str, payload_path: Path, result_path: Path) -> int:
    payload: dict[str, Any] = json.loads(payload_path.read_text(encoding="utf-8"))
    try:
        source_path = Path(payload.pop("source_path"))
        result: list[ParsedChunk] | list[ExtractedMedia]
        if operation == "parse":
            result = parse_document(source_path, **payload)
        elif operation == "media":
            output_dir = Path(payload.pop("output_dir"))
            result = extract_document_media(source_path, output_dir, **payload)
        else:
            raise ValueError("未知隔离任务")
        result_path.write_text(
            json.dumps({"ok": True, "items": [asdict(item) for item in result]}),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        result_path.write_text(
            json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}),
            encoding="utf-8",
        )
        return 1


async def _run(
    operation: Literal["parse", "media"], payload: dict[str, Any]
) -> list[ParsedChunk] | list[ExtractedMedia]:
    with tempfile.TemporaryDirectory(prefix="company-task-") as directory:
        temporary = Path(directory)
        payload_path = temporary / "input.json"
        result_path = temporary / "result.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "company_agent.isolated_tasks",
            operation,
            str(payload_path),
            str(result_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            _, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    await process.wait()
            raise
        if not result_path.exists():
            detail = stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"隔离任务异常退出 ({process.returncode}): {detail}")
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not result_payload["ok"]:
            name, message = result_payload["error"], result_payload["message"]
            if name == "ValueError":
                raise ValueError(message)
            raise RuntimeError(f"{name}: {message}")
        items = result_payload["items"]
        if operation == "parse":
            return [ParsedChunk(**item) for item in items]
        return [
            ExtractedMedia(**{**item, "product_ids": tuple(item["product_ids"])}) for item in items
        ]


async def parse_document_isolated(path: Path, **limits: int) -> list[ParsedChunk]:
    result = await _run("parse", {"source_path": str(path), **limits})
    return cast("list[ParsedChunk]", result)


async def extract_document_media_isolated(
    path: Path, output_dir: Path, **limits: int
) -> list[ExtractedMedia]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="company-media-", dir=output_dir.parent) as stage:
        result = await _run(
            "media",
            {"source_path": str(path), "output_dir": stage, **limits},
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        committed: list[ExtractedMedia] = []
        for item in cast("list[ExtractedMedia]", result):
            destination = output_dir / item.filename
            os.replace(item.stored_path, destination)
            committed.append(replace(item, stored_path=str(destination)))
        return committed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(2)
    raise SystemExit(_worker(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))
