#!/usr/bin/env python3
"""Run the QA matrix with isolated data and executable preconditions."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from company_agent.config import PROJECT_ROOT, Settings
from company_agent.evaluation import (
    QAEvaluator,
    apply_baseline,
    load_baseline,
    load_cases,
    load_checkpoint,
    write_results,
)
from company_agent.ollama_client import OllamaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="隔离执行公司智能助手 QA 用例")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "公司智能助手_全量测试用例.csv",
        help="输入测试用例 CSV（只读）",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge.db",
        help="普通用例使用的只读知识库快照来源",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "公司智能助手_隔离评测结果.csv",
        help="评测结果 CSV",
    )
    parser.add_argument("--baseline", type=Path, help="用于识别新增异常的历史评测结果 CSV")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有输出 CSV 的最后一条用例继续，校验其与当前用例前缀一致",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="仅执行指定用例 ID；可重复传入",
    )
    parser.add_argument("--category", help="仅执行指定问答分类")
    parser.add_argument("--limit", type=int, help="最多执行多少条（调试用）")
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    cases = load_cases(args.cases.expanduser().resolve())
    selected_ids = set(args.case_id)
    if selected_ids:
        cases = [case for case in cases if case.case_id in selected_ids]
        missing = selected_ids - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"未找到用例：{', '.join(sorted(missing))}")
    if args.category:
        cases = [case for case in cases if case.category == args.category]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit 必须大于 0")
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("筛选后没有可执行用例")

    settings = Settings()
    model = OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        model_context=settings.model_context,
        model_timeout_seconds=settings.model_request_timeout_seconds,
        embedding_timeout_seconds=settings.embedding_request_timeout_seconds,
    )
    evaluator = QAEvaluator(settings, args.source_db.expanduser().resolve(), model)
    results = load_checkpoint(args.output.expanduser().resolve(), cases) if args.resume else []
    if results:
        print(f"从检查点恢复：已完成 {len(results)}/{len(cases)} 条", flush=True)
    try:
        for index, case in enumerate(cases[len(results) :], start=len(results) + 1):
            result = await evaluator.evaluate(case)
            results.append(result)
            marker = "PASS" if result.passed else "FAIL"
            print(
                f"[{index}/{len(cases)}] {case.case_id} {marker} "
                f"{result.actual_status}/{result.actual_reason_code} "
                f"({result.duration_ms}ms)",
                flush=True,
            )
            # Keep a recoverable checkpoint even if a later model call is interrupted.
            write_results(args.output.expanduser().resolve(), results)
    finally:
        await model.aclose()

    baseline_compared = args.baseline is not None
    if args.baseline:
        apply_baseline(results, load_baseline(args.baseline.expanduser().resolve()))
    write_results(args.output.expanduser().resolve(), results)

    passed = sum(result.passed for result in results)
    regressions = sum(result.regression == "新增异常" for result in results)
    errors = sum(result.actual_status == "execution_error" for result in results)
    regression_summary = str(regressions) if baseline_compared else "未比较"
    print(
        f"完成：{passed}/{len(results)} 通过，新增异常 {regression_summary}，"
        f"执行异常 {errors}；"
        f"结果：{args.output.expanduser().resolve()}",
        flush=True,
    )
    return 1 if errors or (baseline_compared and regressions) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
