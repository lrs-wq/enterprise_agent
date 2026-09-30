from pathlib import Path

import pytest
from openpyxl import Workbook

from company_agent import document_parser
from company_agent.document_parser import ParsedChunk, parse_document, row_to_text, split_text


def test_split_text_keeps_content_and_respects_limit() -> None:
    text = "第一段。" * 80 + "\n\n" + "第二段。" * 80
    chunks = split_text(text, max_chars=120)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 120 for chunk in chunks)
    assert any("第一段" in chunk for chunk in chunks)
    assert any("第二段" in chunk for chunk in chunks)


def test_parse_xlsx_preserves_sheet_and_row(tmp_path: Path) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "产品表"
    worksheet.append(["产品名称", "卖点"])
    worksheet.append(["宠物营养膏", "补充日常营养"])
    path = tmp_path / "products.xlsx"
    workbook.save(path)

    chunks = parse_document(path)

    assert len(chunks) == 1
    assert chunks[0].sheet == "产品表"
    assert chunks[0].row_start == 2
    assert "宠物营养膏" in chunks[0].content
    assert "补充日常营养" in chunks[0].content


def test_wps_display_image_formula_is_not_indexed_as_text() -> None:
    content = row_to_text(
        ["产品宣传册", "产品名称"],
        ['=DISPIMG("ID_EXAMPLE",1)', "F02饮水机-WIFI款"],
    )

    assert "DISPIMG" not in content
    assert content == "产品名称: F02饮水机-WIFI款"


def test_pdf_uses_ocr_only_for_pages_without_extractable_text(
    tmp_path: Path, monkeypatch: object
) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        pages = [Page("第一页文本"), Page("")]

    requested_pages: list[set[int] | None] = []

    def fake_ocr(
        _path: Path,
        page_numbers: set[int] | None = None,
        *,
        timeout_seconds: int = 240,
    ) -> list[ParsedChunk]:
        assert timeout_seconds == 240
        requested_pages.append(page_numbers)
        return [ParsedChunk(content="第二页 OCR", page=2)]

    monkeypatch.setattr(document_parser, "PdfReader", lambda _path: Reader())  # type: ignore[attr-defined]
    monkeypatch.setattr(document_parser, "parse_pdf_with_macos_vision", fake_ocr)  # type: ignore[attr-defined]
    path = tmp_path / "mixed.pdf"
    path.write_bytes(b"fake")

    chunks = document_parser.parse_pdf(path)

    assert requested_pages == [{2}]
    assert [(chunk.page, chunk.content) for chunk in chunks] == [
        (1, "第一页文本"),
        (2, "第二页 OCR"),
    ]


def test_pdf_page_limit_is_checked_before_extraction(tmp_path: Path, monkeypatch: object) -> None:
    class Page:
        def extract_text(self) -> str:
            return "文本"

    class Reader:
        pages = [Page(), Page()]

    monkeypatch.setattr(document_parser, "PdfReader", lambda _path: Reader())  # type: ignore[attr-defined]
    path = tmp_path / "too-many-pages.pdf"
    path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="页数超过 1 页"):
        document_parser.parse_pdf(path, max_pages=1)


def test_xlsx_row_and_uncompressed_size_limits(tmp_path: Path) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["产品名称"])
    worksheet.append(["产品 A"])
    worksheet.append(["产品 B"])
    path = tmp_path / "limited.xlsx"
    workbook.save(path)

    with pytest.raises(ValueError, match="数据行超过 2 行"):
        parse_document(path, max_spreadsheet_rows=2)
    with pytest.raises(ValueError, match="解压后体积超过"):
        parse_document(path, max_archive_uncompressed_bytes=1)
