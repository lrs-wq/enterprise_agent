#!/usr/bin/env python3
"""Measure a fixed set of real model questions without changing the source database."""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import tempfile
import time
from pathlib import Path

from company_agent.config import PROJECT_ROOT, Settings
from company_agent.database import KnowledgeRepository
from company_agent.evaluation import clone_database, load_cases
from company_agent.ollama_client import OllamaClient
from company_agent.schemas import ChatRequest
from company_agent.services import AgentService

DEFAULT_CASE_IDS = (
    "QA-001",
    "QA-002",
    "QA-004",
    "QA-009",
    "QA-012",
    "QA-021",
    "QA-022",
    "QA-027",
    "QA-030",
    "QA-040",
)
FIELDS = (
    "run",
    "case_id",
    "question",
    "status",
    "reason_code",
    "evidence_status",
    "wall_ms",
    "topic_check_ms",
    "retrieval_ms",
    "generation_ms",
    "repair_ms",
    "cache_hit",
    "policy_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定公司/产品题集的真实模型延迟基准")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "公司智能助手_全量测试用例.csv",
    )
    parser.add_argument("--source-db", type=Path, default=PROJECT_ROOT / "data" / "knowledge.db")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


async def run() -> None:
    args = parse_args()
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat 必须大于 0，warmup 不能为负")
    case_ids = tuple(args.case_id) or DEFAULT_CASE_IDS
    by_id = {case.case_id: case for case in load_cases(args.cases.resolve())}
    missing = set(case_ids) - set(by_id)
    if missing:
        raise ValueError(f"未找到基准用例：{', '.join(sorted(missing))}")
    selected = [by_id[case_id] for case_id in case_ids]
    sequence = [(run_number, case) for run_number in range(1, args.repeat + 1) for case in selected]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, str]] = []
    if args.resume:
        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("已有基准文件字段不一致")
            completed = list(reader)
        if len(completed) > len(sequence):
            raise ValueError("已有基准记录多于当前任务")
        for row, (run_number, case) in zip(completed, sequence, strict=False):
            if (row["run"], row["case_id"], row["question"]) != (
                str(run_number),
                case.case_id,
                case.question,
            ):
                raise ValueError("已有基准记录与当前题集不一致")

    settings = Settings()
    with tempfile.TemporaryDirectory(prefix="company-latency-") as temporary:
        isolated = Settings(data_dir=Path(temporary), answer_cache_enabled=not args.disable_cache)
        isolated.prepare_directories()
        clone_database(args.source_db.resolve(), isolated.database_path)
        repository = KnowledgeRepository(isolated.database_path)
        repository.initialize()
        model = OllamaClient(
            base_url=settings.ollama_base_url,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            model_context=settings.model_context,
            model_timeout_seconds=settings.model_request_timeout_seconds,
            embedding_timeout_seconds=settings.embedding_request_timeout_seconds,
        )
        service = AgentService(isolated, repository, model)
        try:
            for _ in range(args.warmup):
                await service.chat(ChatRequest(question=selected[0].question))
            with output.open(
                "a" if args.resume else "w",
                encoding="utf-8-sig" if not args.resume else "utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if not args.resume:
                    writer.writeheader()
                for index, (run_number, case) in enumerate(
                    sequence[len(completed) :], start=len(completed) + 1
                ):
                    started = time.perf_counter()
                    response = await service.chat(ChatRequest(question=case.question))
                    row = {
                        "run": run_number,
                        "case_id": case.case_id,
                        "question": case.question,
                        "status": response.status,
                        "reason_code": response.decision.reason_code,
                        "evidence_status": response.evidence_status,
                        "wall_ms": round((time.perf_counter() - started) * 1000),
                        "topic_check_ms": int(response.metrics.get("topic_check_ms", 0)),
                        "retrieval_ms": int(response.metrics.get("retrieval_ms", 0)),
                        "generation_ms": int(response.metrics.get("generation_ms", 0)),
                        "repair_ms": int(response.metrics.get("repair_ms", 0)),
                        "cache_hit": int(response.metrics.get("cache_hit", 0)),
                        "policy_version": response.decision.policy_version,
                    }
                    writer.writerow(row)
                    handle.flush()
                    completed.append({key: str(value) for key, value in row.items()})
                    print(
                        f"[{index}/{len(sequence)}] {case.case_id} "
                        f"{response.status} {row['wall_ms']}ms",
                        flush=True,
                    )
        finally:
            await model.aclose()

    durations = [int(row["wall_ms"]) for row in completed]
    print(
        f"完成 {len(durations)} 次；P50={round(statistics.median(durations))}ms，"
        f"P95={percentile(durations, 0.95)}ms；结果：{output}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(run())
