from company_agent.product_identity import (
    extract_product_facets,
    extract_product_identities,
    product_facets_match,
)


def test_same_model_across_categories_and_variants_has_stable_distinct_ids() -> None:
    text = """
    CKIN PET
    F02饮水机-WIFI款
    F02 Water Dispenser - Basic Version
    F02喂食器-视频款
    F02 Pet Feeder - WiFi Version
    F02喂食器-按键款
    """

    identities = extract_product_identities(text)
    stable_ids = {identity.stable_id for identity in identities}

    assert stable_ids == {
        "ckin-pet:water-dispenser:F02:wifi",
        "ckin-pet:water-dispenser:F02:basic",
        "ckin-pet:pet-feeder:F02:video",
        "ckin-pet:pet-feeder:F02:wifi",
        "ckin-pet:pet-feeder:F02:button",
    }


def test_explicit_category_and_variant_must_match_media_candidate() -> None:
    requested = extract_product_facets("CKIN PET F02 视频款喂食器")
    feeder = extract_product_facets("CKIN PET F02 Pet Feeder - Video Version")
    fountain = extract_product_facets("CKIN PET F02 Water Dispenser - WiFi Version")

    assert product_facets_match(requested, feeder) is True
    assert product_facets_match(requested, fountain) is False
