#!/usr/bin/env python3
"""Export the executable QA result CSV as a readable question-answer document."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from company_agent.config import PROJECT_ROOT

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "公司智能助手_第二阶段全量回归_2026-09-22.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "公司智能助手_修复后问答结果_2026-09-23.md"

STATUS_LABELS = {
    "grounded": "有据回答",
    "general_knowledge": "模型通用知识",
    "partial_answer": "仅回答允许部分",
    "out_of_scope": "服务范围说明",
    "clarification_needed": "需要补充信息",
    "insufficient_evidence": "证据不足",
    "service_unavailable": "服务暂不可用",
    "conversational": "自然对话",
}

EVIDENCE_LABELS = {
    "supported": "已支持",
    "partial": "部分支持",
    "missing": "缺少证据",
    "not_applicable": "无需检索",
    "model_knowledge": "模型自身知识",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 QA 回归 CSV 导出为问题—回复 Markdown")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="回归结果 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 Markdown")
    parser.add_argument(
        "--media-base-url",
        default="http://127.0.0.1:8000",
        help="图片相对 URL 的访问前缀",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("回归结果 CSV 为空")
    identifiers = [row.get("用例ID", "").strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("存在空用例 ID")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("用例 ID 不唯一")
    return rows


def quote(text: str) -> list[str]:
    lines = text.strip().splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def clean_heading(text: str, fallback: str) -> str:
    value = re.sub(r"\s+", " ", text).strip().lstrip("#").strip()
    return value or fallback


def normalize_text(text: str) -> str:
    return re.sub(r"<br\s*/?>", "  \n", text.strip(), flags=re.IGNORECASE)


def source_location(item: dict[str, Any]) -> str:
    parts: list[str] = []
    if item.get("page"):
        parts.append(f"第 {item['page']} 页")
    if item.get("sheet"):
        parts.append(f"工作表 {item['sheet']}")
    if item.get("row_start"):
        row = str(item["row_start"])
        if item.get("row_end") and item["row_end"] != item["row_start"]:
            row = f"{item['row_start']}-{item['row_end']}"
        parts.append(f"第 {row} 行")
    return "，".join(parts) or "位置未标注"


def media_url(base_url: str, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def render_answer(payload: dict[str, Any], media_base_url: str) -> list[str]:
    lines: list[str] = []
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        error = str(payload.get("error") or "请求未形成结构化回复")
        detail = str(payload.get("detail") or "")
        error_title = {
            "timeout": "生成请求超时",
            "unavailable": "服务暂不可用",
        }.get(error.lower(), error)
        lines.extend((f"#### {clean_heading(error_title, '请求未完成')}", ""))
        if detail:
            lines.append(detail)
        return lines

    title = clean_heading(str(answer.get("title") or "回复"), "回复")
    summary = normalize_text(str(answer.get("summary") or ""))
    lines.extend((f"#### {title}", "", summary or "（无正文）"))

    for section in answer.get("sections") or []:
        if not isinstance(section, dict):
            continue
        heading = clean_heading(str(section.get("heading") or "详细说明"), "详细说明")
        content = normalize_text(str(section.get("content") or ""))
        if not content:
            continue
        lines.extend(("", f"##### {heading}", "", content))

    reason = normalize_text(str(answer.get("refusal_reason") or ""))
    compact_summary = re.sub(r"\s+", " ", summary).strip()
    compact_reason = re.sub(r"\s+", " ", reason).strip()
    if reason and compact_reason not in compact_summary and compact_summary not in compact_reason:
        lines.extend(("", "**补充说明**", "", *quote(reason)))

    images = [item for item in payload.get("images") or [] if isinstance(item, dict)]
    if images:
        lines.extend(("", "**图片**", ""))
        for image in images:
            caption = str(image.get("caption") or image.get("alt_text") or "产品图片")
            source_id = str(image.get("source_id") or "").strip()
            source_suffix = f"，来源 {source_id}" if source_id else ""
            url = media_url(media_base_url, str(image.get("url") or ""))
            lines.append(
                f"- {caption}（{source_location(image)}{source_suffix}）：[查看图片]({url})"
            )

    sources = [item for item in payload.get("sources") or [] if isinstance(item, dict)]
    if sources:
        lines.extend(("", "**来源**", ""))
        for source in sources:
            kind = "图片来源" if source.get("source_kind") == "media" else "文字来源"
            lines.append(
                f"- {source.get('source_id', '-')}（{kind}）："
                f"{source.get('document_title', '未命名资料')}，{source_location(source)}"
            )

    follow_ups = [str(item).strip() for item in payload.get("follow_up_questions") or []]
    follow_ups = [item for item in follow_ups if item]
    if follow_ups:
        lines.extend(("", "**可继续询问**", ""))
        lines.extend(f"- {item}" for item in follow_ups)
    return lines


def render_document(rows: list[dict[str, str]], source: Path, media_base_url: str) -> str:
    result_counts = Counter(row.get("测试结果", "") for row in rows)
    status_counts = Counter(row.get("实际状态", "") for row in rows)
    regressions = sum(row.get("回归结果") == "新增异常" for row in rows)
    errors = sum(bool(row.get("执行错误", "").strip()) for row in rows)

    lines = [
        "# 公司智能助手修复后问答结果",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        f"> 数据来源：`{source.name}`",
        f"> 共 {len(rows)} 条问答；通过 {result_counts.get('通过', 0)} 条；"
        f"新增异常 {regressions} 条；执行异常 {errors} 条。",
        "> 说明：`service_unavailable` 用例为主动故障注入验证，返回服务故障是预期结果，"
        "不表示本次评测运行失败。",
        "",
        "## 结果概览",
        "",
        "| 回答状态 | 数量 |",
        "| --- | ---: |",
    ]
    for status, count in status_counts.items():
        lines.append(f"| {STATUS_LABELS.get(status, status)}（`{status}`） | {count} |")

    current_category = ""
    for row in rows:
        category = row.get("问答分类", "未分类").strip() or "未分类"
        if category != current_category:
            lines.extend(("", f"## {category}", ""))
            current_category = category

        case_id = row.get("用例ID", "").strip()
        scenario = clean_heading(row.get("测试场景", ""), "未命名场景")
        lines.extend((f"### {case_id} · {scenario}", "", "**问题**", ""))
        lines.extend(quote(row.get("用户询问语句", "")))

        precondition = row.get("前置/对话上下文", "").strip()
        has_meaningful_precondition = precondition not in {"", "无历史", "无历史。"}
        if has_meaningful_precondition and category in {"多轮问答与指代", "服务异常与降级"}:
            label = "对话背景" if category == "多轮问答与指代" else "验证条件"
            lines.extend(("", f"**{label}**", "", *quote(precondition)))

        lines.extend(("", "**回复**", ""))
        raw_answer = row.get("实际模型回答", "").strip()
        try:
            payload = json.loads(raw_answer)
        except json.JSONDecodeError:
            payload = {"answer": {"title": "回复", "summary": raw_answer}}
        if not isinstance(payload, dict):
            payload = {"answer": {"title": "回复", "summary": str(payload)}}
        lines.extend(render_answer(payload, media_base_url))

        status = row.get("实际状态", "").strip()
        evidence = row.get("实际证据状态", "").strip()
        test_result = row.get("测试结果", "").strip()
        lines.extend(
            (
                "",
                "**验证结果**："
                f"{test_result or '未记录'} · "
                f"{STATUS_LABELS.get(status, status or '未知状态')} · "
                f"{EVIDENCE_LABELS.get(evidence, evidence or '未知证据状态')}",
                "",
                "---",
                "",
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    rows = load_rows(input_path)
    markdown = render_document(rows, input_path, args.media_base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"已生成 {len(rows)} 条问答：{output_path}")


if __name__ == "__main__":
    main()
