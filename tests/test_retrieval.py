from company_agent.database import pack_embedding
from company_agent.retrieval import build_context, lexical_score, retrieve


def make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    embedding: list[float],
) -> dict[str, object]:
    return {
        "id": chunk_id,
        "document_id": document_id,
        "document_title": f"文档-{document_id}",
        "filename": f"{document_id}.pdf",
        "content": content,
        "embedding": pack_embedding(embedding),
        "page": 2,
        "sheet": None,
        "row_start": None,
        "row_end": None,
    }


def test_lexical_score_understands_chinese_bigrams() -> None:
    relevant = lexical_score("宠物营养", "这是一款宠物日常营养产品")
    irrelevant = lexical_score("宠物营养", "公司联系地址")

    assert relevant > irrelevant


def test_retrieve_ranks_and_builds_citable_context() -> None:
    chunks = [
        make_chunk("c1", "d1", "宠物营养产品介绍", [1.0, 0.0]),
        make_chunk("c2", "d2", "公司发展历程", [0.0, 1.0]),
    ]

    results = retrieve("营养产品", [1.0, 0.0], chunks, top_k=2)
    context, used_results = build_context(results)

    assert results[0].source.chunk_id == "c1"
    assert results[0].source.source_id == "S1"
    assert "[S1]" in context
    assert "第 2 页" in context
    assert used_results == results


def test_build_context_returns_only_results_actually_in_model_window() -> None:
    chunks = [
        make_chunk("c1", "d1", "第一条证据", [1.0, 0.0]),
        make_chunk("c2", "d2", "第二条证据", [0.9, 0.1]),
    ]
    results = retrieve("证据", [1.0, 0.0], chunks, top_k=2)

    context, used_results = build_context(results, max_chars=80)

    assert len(used_results) == 1
    assert "[S1]" in context
    assert "[S2]" not in context
