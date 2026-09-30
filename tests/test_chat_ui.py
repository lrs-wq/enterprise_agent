from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_chat_ui_folds_technical_metadata_and_distinguishes_scope_only() -> None:
    script = (PROJECT_ROOT / "static" / "chat.js").read_text(encoding="utf-8")

    assert "<summary>技术详情</summary>" in script
    assert "scope_only: '仅说明服务范围'" in script
    assert "verified_extractive: '已从审核资料提取回答'" in script
    assert "shouldShowLimitation(data)" in script


def test_chat_ui_renders_media_citation_ids() -> None:
    script = (PROJECT_ROOT / "static" / "chat.js").read_text(encoding="utf-8")

    assert "([SM])" in script
    assert "图片来源" in script
