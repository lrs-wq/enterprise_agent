import math
import re
from dataclasses import dataclass

import numpy as np

from company_agent.database import unpack_embedding
from company_agent.schemas import SourceRef


@dataclass(frozen=True)
class SearchResult:
    source: SourceRef
    content: str


def lexical_tokens(text: str) -> set[str]:
    normalized = text.lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9_.+-]*", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    characters = set(chinese)
    bigrams = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}
    return words | characters | bigrams


def lexical_score(query: str, content: str) -> float:
    query_tokens = lexical_tokens(query)
    if not query_tokens:
        return 0.0
    content_tokens = lexical_tokens(content)
    return len(query_tokens & content_tokens) / len(query_tokens)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(left, right) / denominator)


def retrieve(
    query: str,
    query_embedding: list[float],
    chunks: list[dict[str, object]],
    top_k: int,
) -> list[SearchResult]:
    query_vector = np.asarray(query_embedding, dtype=np.float32)
    scored: list[tuple[float, dict[str, object]]] = []
    for chunk in chunks:
        raw_embedding = chunk["embedding"]
        if not isinstance(raw_embedding, bytes):
            continue
        vector_score = cosine_similarity(query_vector, unpack_embedding(raw_embedding))
        keyword_score = lexical_score(query, str(chunk["content"]))
        combined_score = (0.85 * max(vector_score, 0.0)) + (0.15 * keyword_score)
        scored.append((combined_score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[SearchResult] = []
    per_document: dict[str, int] = {}
    for score, chunk in scored:
        document_id = str(chunk["document_id"])
        if per_document.get(document_id, 0) >= 3:
            continue
        source_id = f"S{len(selected) + 1}"
        content = str(chunk["content"])
        page = chunk["page"] if isinstance(chunk["page"], int) else None
        sheet = chunk["sheet"] if isinstance(chunk["sheet"], str) else None
        row_start = chunk["row_start"] if isinstance(chunk["row_start"], int) else None
        row_end = chunk["row_end"] if isinstance(chunk["row_end"], int) else None
        raw_source_version = chunk.get("source_version")
        source_version = raw_source_version if isinstance(raw_source_version, int) else None
        selected.append(
            SearchResult(
                source=SourceRef(
                    source_id=source_id,
                    chunk_id=str(chunk["id"]),
                    document_id=document_id,
                    document_title=str(chunk["document_title"]),
                    filename=str(chunk["filename"]),
                    page=page,
                    sheet=sheet,
                    row_start=row_start,
                    row_end=row_end,
                    score=round(score, 4),
                    excerpt=content[:320],
                    source_version=source_version,
                ),
                content=content,
            )
        )
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def build_context(
    results: list[SearchResult], max_chars: int = 11_000
) -> tuple[str, list[SearchResult]]:
    blocks: list[str] = []
    used_results: list[SearchResult] = []
    used_chars = 0
    for result in results:
        source = result.source
        location_parts: list[str] = []
        if source.page is not None:
            location_parts.append(f"第 {source.page} 页")
        if source.sheet:
            location_parts.append(f"工作表 {source.sheet}")
        if source.row_start is not None:
            row_label = str(source.row_start)
            if source.row_end and source.row_end != source.row_start:
                row_label = f"{source.row_start}-{source.row_end}"
            location_parts.append(f"第 {row_label} 行")
        location = "，".join(location_parts) or "位置未标注"
        block = (
            f"[{source.source_id}] 文档: {source.document_title}；文件: {source.filename}；"
            f"位置: {location}\n{result.content}"
        )
        separator_chars = 7 if blocks else 0
        if used_chars + separator_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_results.append(result)
        used_chars += separator_chars + len(block)
    return "\n\n---\n\n".join(blocks), used_results
