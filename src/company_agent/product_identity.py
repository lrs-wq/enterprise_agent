"""Deterministic product identity extraction shared by text and media routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MODEL_PATTERN = re.compile(r"(?<![A-Z0-9])([A-Z]{1,6}[- ]?\d{2,6})(?![A-Z0-9])")

_BRAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ckin-pet", re.compile(r"\bCKIN\s*PET\b|\bCKINPET\b|\bCKIN\b", re.IGNORECASE)),
    ("chongqin", re.compile(r"\bCHONGQIN\b|宠沁", re.IGNORECASE)),
    ("gougelila", re.compile(r"\bGOUGELILA\b|狗哥里啦", re.IGNORECASE)),
)

_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("water-dispenser", re.compile(r"饮水机|喂水器|water\s*(?:fountain|dispenser)", re.I)),
    ("pet-feeder", re.compile(r"喂食器|pet\s*feeder|\bfeeder\b", re.I)),
    ("nail-grinder", re.compile(r"磨甲器|nail\s*grinder", re.I)),
    ("pet-groomer", re.compile(r"剃毛器|理毛器|pet\s*groomer|clipper", re.I)),
    ("pet-dryer", re.compile(r"吹吸机|吹风机|烘干(?:机|箱)|pet\s*dryer", re.I)),
    ("litter-box", re.compile(r"猫砂盆|litter\s*box", re.I)),
    ("dog-toilet", re.compile(r"狗厕所|dog\s*toilet", re.I)),
    ("playpen", re.compile(r"围栏|playpen", re.I)),
    ("pet-cage", re.compile(r"宠物笼|pet\s*cage", re.I)),
    ("training-pad", re.compile(r"尿垫|training\s*pads?", re.I)),
    ("toy", re.compile(r"玩具|\btoys?\b", re.I)),
)

_VARIANT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("video", re.compile(r"视频款|video\s*version", re.I)),
    ("wifi", re.compile(r"wi[ -]?fi\s*款|wifi\s*version", re.I)),
    ("basic", re.compile(r"基础款|basic\s*version", re.I)),
    ("button", re.compile(r"按键款|button\s*version", re.I)),
    ("cordless", re.compile(r"无线款|cordless\s*version", re.I)),
    ("corded", re.compile(r"插线款|corded\s*version", re.I)),
    ("dual-bin", re.compile(r"双仓(?:款)?|dual[- ]?bin", re.I)),
)


@dataclass(frozen=True)
class ProductFacets:
    brands: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    models: frozenset[str] = frozenset()
    variants: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ProductIdentity:
    brand: str
    category: str
    model: str
    variant: str

    @property
    def stable_id(self) -> str:
        """Return an opaque, deterministic ID safe to persist or compare."""

        return ":".join((self.brand, self.category, self.model, self.variant))


def extract_product_facets(text: str) -> ProductFacets:
    brands = frozenset(name for name, pattern in _BRAND_PATTERNS if pattern.search(text))
    categories = frozenset(name for name, pattern in _CATEGORY_PATTERNS if pattern.search(text))
    models = frozenset(
        match.replace(" ", "").upper() for match in _MODEL_PATTERN.findall(text.upper())
    )
    variants = frozenset(name for name, pattern in _VARIANT_PATTERNS if pattern.search(text))
    return ProductFacets(
        brands=brands,
        categories=categories,
        models=models,
        variants=variants,
    )


def extract_product_identities(
    text: str,
    *,
    default_brand: str = "ckin-pet",
) -> tuple[ProductIdentity, ...]:
    """Split row/page text into explicit brand-category-model-variant identities.

    Product sheets often keep several variants in one row. Each header line is parsed
    separately so, for example, F02 feeder/video and F02 feeder/Wi-Fi remain distinct
    from F02 water-dispenser/Wi-Fi.
    """

    global_facets = extract_product_facets(text)
    brand = next(iter(global_facets.brands), default_brand)
    identities: list[ProductIdentity] = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        facets = extract_product_facets(line)
        if not facets.models or not facets.categories:
            continue
        variants = facets.variants or frozenset({"standard"})
        line_brand = next(iter(facets.brands), brand)
        for model in sorted(facets.models):
            for category in sorted(facets.categories):
                for variant in sorted(variants):
                    identities.append(
                        ProductIdentity(
                            brand=line_brand,
                            category=category,
                            model=model,
                            variant=variant,
                        )
                    )

    if not identities and len(global_facets.categories) == 1 and global_facets.models:
        variants = global_facets.variants or frozenset({"standard"})
        for model in sorted(global_facets.models):
            for variant in sorted(variants):
                identities.append(
                    ProductIdentity(
                        brand=brand,
                        category=next(iter(global_facets.categories)),
                        model=model,
                        variant=variant,
                    )
                )

    return tuple(dict.fromkeys(identities))


def product_facets_match(requested: ProductFacets, candidate: ProductFacets) -> bool:
    """Require every facet explicitly requested by the user to match the asset."""

    return all(
        not expected or bool(expected & actual)
        for expected, actual in (
            (requested.brands, candidate.brands),
            (requested.categories, candidate.categories),
            (requested.models, candidate.models),
            (requested.variants, candidate.variants),
        )
    )
