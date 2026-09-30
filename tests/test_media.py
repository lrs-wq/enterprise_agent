from pathlib import Path

import pytest

from company_agent.media import extract_document_media


def test_wps_dispimg_assets_keep_product_row_and_original_pixels(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "uploads" / "Ckin Pet宣传册.xls"
    if not source.exists():
        pytest.skip("本地 WPS 宣传册样本不存在")

    assets = extract_document_media(source, tmp_path / "media")
    f02 = [asset for asset in assets if asset.row_start == 9 and "F02" in asset.product_ids]

    assert len(assets) == 19
    assert len(f02) == 1
    assert f02[0].caption == "F02饮水机-WIFI款"
    assert f02[0].mime_type == "image/png"
    assert f02[0].width >= 500
    assert f02[0].height >= 500
    assert Path(f02[0].stored_path).is_file()
