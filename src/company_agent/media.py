from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import pymupdf
import xlrd
import xlrd.compdoc
from openpyxl import load_workbook
from PIL import Image, UnidentifiedImageError

from company_agent.document_parser import normalize_text, validate_office_archive

_DISPIMG_PATTERN = re.compile(r'DISPIMG\(\s*"([^"]+)"', re.IGNORECASE)
_PRODUCT_CODE_PATTERN = re.compile(r"(?<![A-Z0-9])([A-Z]{1,6}[- ]?\d{2,6})(?![A-Z0-9])")
_PRODUCT_WORDS = (
    "饮水机",
    "喂食器",
    "猫砂",
    "猫砂盆",
    "狗厕所",
    "围栏",
    "宠物笼",
    "尿垫",
    "玩具",
    "磨甲器",
    "剃毛器",
    "吹吸机",
    "吹风机",
    "烘干箱",
    "water fountain",
    "feeder",
    "litter",
    "toilet",
    "playpen",
    "pet cage",
)
_GENERIC_CAPTIONS = {
    "英文",
    "contents",
    "产品信息",
    "product information",
    "主推品",
}


@dataclass(frozen=True)
class ExtractedMedia:
    image_key: str
    filename: str
    stored_path: str
    mime_type: str
    sha256: str
    width: int
    height: int
    source_kind: str
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    cell_ref: str | None = None
    caption: str = ""
    alt_text: str = ""
    product_ids: tuple[str, ...] = ()
    search_text: str = ""


def _column_name(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _first_line(value: object) -> str:
    text = normalize_text(value)
    for line in text.splitlines():
        candidate = line.strip(" ：:|-/")
        if candidate and candidate.lower() not in _GENERIC_CAPTIONS:
            return candidate[:160]
    return ""


def _extract_product_ids(text: str) -> tuple[str, ...]:
    values = [match.replace(" ", "").upper() for match in _PRODUCT_CODE_PATTERN.findall(text)]
    return tuple(dict.fromkeys(values))


def _is_product_image(caption: str, search_text: str) -> bool:
    value = f"{caption}\n{search_text}".lower()
    return bool(_extract_product_ids(value)) or any(word in value for word in _PRODUCT_WORDS)


def _image_properties(payload: bytes) -> tuple[str, str, int, int] | None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError):
        return None
    formats = {
        "PNG": ("png", "image/png"),
        "JPEG": ("jpg", "image/jpeg"),
        "WEBP": ("webp", "image/webp"),
        "GIF": ("gif", "image/gif"),
        "BMP": ("bmp", "image/bmp"),
        "TIFF": ("tiff", "image/tiff"),
    }
    suffix, mime_type = formats.get(
        image_format, (image_format.lower(), "application/octet-stream")
    )
    if not suffix or mime_type == "application/octet-stream":
        return None
    return suffix, mime_type, width, height


def _store_image(
    payload: bytes,
    output_dir: Path,
    image_key: str,
) -> tuple[str, str, str, int, int] | None:
    properties = _image_properties(payload)
    if properties is None:
        return None
    suffix, mime_type, width, height = properties
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", image_key).strip("-")[:150]
    filename = f"{safe_key or hashlib.sha256(payload).hexdigest()[:16]}.{suffix}"
    destination = output_dir / filename
    if not destination.exists() or destination.read_bytes() != payload:
        destination.write_bytes(payload)
    return filename, mime_type, str(destination), width, height


def _wps_image_map(archive: ZipFile) -> dict[str, str]:
    try:
        cell_images = ET.fromstring(archive.read("xl/cellImages.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/cellImages.xml.rels"))
    except (KeyError, ET.ParseError):
        return {}
    rel_targets = {
        relation.attrib.get("Id", ""): relation.attrib.get("Target", "")
        for relation in relationships
    }
    result: dict[str, str] = {}
    for cell_image in cell_images:
        image_name = ""
        relationship_id = ""
        for element in cell_image.iter():
            local_name = element.tag.rsplit("}", 1)[-1]
            if local_name == "cNvPr":
                image_name = element.attrib.get("name", "")
            elif local_name == "blip":
                relationship_id = next(
                    (
                        value
                        for key, value in element.attrib.items()
                        if key.rsplit("}", 1)[-1] == "embed"
                    ),
                    "",
                )
        target = rel_targets.get(relationship_id, "")
        if image_name and target:
            result[image_name] = f"xl/{target.lstrip('/')}"
    return result


def _xls_cell_context(
    sheet: xlrd.sheet.Sheet, row_index: int, column_index: int
) -> tuple[str, str]:
    row_values = sheet.row_values(row_index)
    right_values = [
        value
        for value in row_values[column_index + 1 :]
        if normalize_text(value) and not _DISPIMG_PATTERN.search(normalize_text(value))
    ]
    all_values = [
        normalize_text(value)
        for value in row_values
        if normalize_text(value) and not _DISPIMG_PATTERN.search(normalize_text(value))
    ]
    caption = next((_first_line(value) for value in right_values if _first_line(value)), "")
    if not caption:
        caption = next((_first_line(value) for value in all_values if _first_line(value)), "")
    return caption, "\n".join(all_values)[:12_000]


def _extract_wps_xls(path: Path, output_dir: Path) -> list[ExtractedMedia]:
    raw_file = path.read_bytes()
    compound = xlrd.compdoc.CompDoc(raw_file, logfile=io.StringIO(), DEBUG=0)
    try:
        cell_image_data = compound.get_named_stream("ETCellImageData")
    except Exception:  # xlrd exposes format-specific errors without one common base class.
        return []

    if not isinstance(cell_image_data, bytes):
        return []

    try:
        archive = ZipFile(io.BytesIO(cell_image_data))
    except BadZipFile:
        return []

    workbook = xlrd.open_workbook(str(path), on_demand=True)
    extracted: list[ExtractedMedia] = []
    seen_image_keys: set[str] = set()
    try:
        with archive:
            image_map = _wps_image_map(archive)
            for sheet in workbook.sheets():
                for row_index in range(sheet.nrows):
                    for column_index, raw_value in enumerate(sheet.row_values(row_index)):
                        match = _DISPIMG_PATTERN.search(normalize_text(raw_value))
                        if match is None:
                            continue
                        image_id = match.group(1)
                        archive_path = image_map.get(image_id)
                        if archive_path is None or image_id in seen_image_keys:
                            continue
                        try:
                            payload = archive.read(archive_path)
                        except KeyError:
                            continue
                        image_key = f"wps-{image_id.lower()}"
                        stored = _store_image(payload, output_dir, image_key)
                        if stored is None:
                            continue
                        filename, mime_type, stored_path, width, height = stored
                        caption, search_text = _xls_cell_context(sheet, row_index, column_index)
                        row_number = row_index + 1
                        cell_ref = f"{_column_name(column_index)}{row_number}"
                        alt_text = caption or f"{sheet.name} 第 {row_number} 行图片"
                        extracted.append(
                            ExtractedMedia(
                                image_key=image_key,
                                filename=filename,
                                stored_path=stored_path,
                                mime_type=mime_type,
                                sha256=hashlib.sha256(payload).hexdigest(),
                                width=width,
                                height=height,
                                source_kind=(
                                    "product_image"
                                    if _is_product_image(caption, search_text)
                                    else "document_image"
                                ),
                                sheet=sheet.name,
                                row_start=row_number,
                                row_end=row_number,
                                cell_ref=cell_ref,
                                caption=caption,
                                alt_text=alt_text,
                                product_ids=_extract_product_ids(search_text),
                                search_text=search_text,
                            )
                        )
                        seen_image_keys.add(image_id)
    finally:
        workbook.release_resources()
    return extracted


def _xlsx_cell_context(worksheet: Any, row_number: int, column_index: int) -> tuple[str, str]:
    values: list[str] = []
    right_values: list[str] = []
    for cell in worksheet[row_number]:
        text = normalize_text(cell.value) if cell.value is not None else ""
        if not text or _DISPIMG_PATTERN.search(text):
            continue
        values.append(text)
        if int(cell.column) > column_index:
            right_values.append(text)
    caption = next((_first_line(value) for value in right_values if _first_line(value)), "")
    if not caption:
        caption = next((_first_line(value) for value in values if _first_line(value)), "")
    return caption, "\n".join(values)[:12_000]


def _extract_xlsx(path: Path, output_dir: Path) -> list[ExtractedMedia]:
    workbook = load_workbook(path, read_only=False, data_only=False)
    extracted: list[ExtractedMedia] = []
    try:
        for worksheet in workbook.worksheets:
            images: list[Any] = list(getattr(worksheet, "_images", []))
            for position, image in enumerate(images, start=1):
                try:
                    payload = image._data()
                    marker = image.anchor._from
                    row_number = int(marker.row) + 1
                    column_number = int(marker.col) + 1
                except (AttributeError, OSError, TypeError, ValueError):
                    continue
                image_key = f"xlsx-{worksheet.title}-{row_number}-{column_number}-{position}"
                stored = _store_image(payload, output_dir, image_key)
                if stored is None:
                    continue
                filename, mime_type, stored_path, width, height = stored
                caption, search_text = _xlsx_cell_context(worksheet, row_number, column_number)
                extracted.append(
                    ExtractedMedia(
                        image_key=image_key,
                        filename=filename,
                        stored_path=stored_path,
                        mime_type=mime_type,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        width=width,
                        height=height,
                        source_kind=(
                            "product_image"
                            if _is_product_image(caption, search_text)
                            else "document_image"
                        ),
                        sheet=worksheet.title,
                        row_start=row_number,
                        row_end=row_number,
                        cell_ref=f"{_column_name(column_number - 1)}{row_number}",
                        caption=caption,
                        alt_text=caption or f"{worksheet.title} 第 {row_number} 行图片",
                        product_ids=_extract_product_ids(search_text),
                        search_text=search_text,
                    )
                )
    finally:
        workbook.close()
    return extracted


def _extract_pdf(
    path: Path,
    output_dir: Path,
    *,
    max_pages: int,
) -> list[ExtractedMedia]:
    document = pymupdf.open(path)  # type: ignore[no-untyped-call]
    extracted: list[ExtractedMedia] = []
    try:
        if document.page_count > max_pages:
            raise ValueError(f"PDF 页数超过 {max_pages} 页限制")
        matrix = pymupdf.Matrix(1.35, 1.35)  # type: ignore[no-untyped-call]
        for page_index in range(document.page_count):
            page = document.load_page(page_index)  # type: ignore[no-untyped-call]
            page_number = page_index + 1
            text = normalize_text(page.get_text("text"))
            caption = _first_line(text) or f"{path.stem} 第 {page_number} 页"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            payload = pixmap.tobytes("jpeg", jpg_quality=84)
            image_key = f"pdf-page-{page_number}"
            stored = _store_image(payload, output_dir, image_key)
            if stored is None:
                continue
            filename, mime_type, stored_path, width, height = stored
            extracted.append(
                ExtractedMedia(
                    image_key=image_key,
                    filename=filename,
                    stored_path=stored_path,
                    mime_type=mime_type,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    width=width,
                    height=height,
                    source_kind="page_preview",
                    page=page_number,
                    caption=caption,
                    alt_text=f"{path.stem} 第 {page_number} 页：{caption}",
                    product_ids=_extract_product_ids(text),
                    search_text=text[:12_000],
                )
            )
    finally:
        document.close()  # type: ignore[no-untyped-call]
    return extracted


def extract_document_media(
    path: Path,
    output_dir: Path,
    *,
    max_pdf_pages: int = 300,
    max_archive_uncompressed_bytes: int = 256 * 1024 * 1024,
    max_archive_members: int = 5000,
) -> list[ExtractedMedia]:
    """Extract only images physically present in the uploaded source document."""

    output_dir.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    if extension == ".xls":
        return _extract_wps_xls(path, output_dir)
    if extension == ".xlsx":
        validate_office_archive(
            path,
            max_uncompressed_bytes=max_archive_uncompressed_bytes,
            max_members=max_archive_members,
        )
        return _extract_xlsx(path, output_dir)
    if extension == ".pdf":
        return _extract_pdf(path, output_dir, max_pages=max_pdf_pages)
    return []
