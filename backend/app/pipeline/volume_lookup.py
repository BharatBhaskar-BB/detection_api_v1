"""Volume and weight lookup for common moving inventory items.

Uses industry-standard moving estimates. Items keyed by (name, size).
Fallback: compute from LLM-estimated dimensions.
"""

from __future__ import annotations

# ── Lookup Table ──
# Each entry: (volume_cuft, weight_lbs)
# Sources: industry moving cube sheets, AMSA standards
_VOLUME_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    # ── Seating ──
    ("dining chair", "small"):       (5, 15),
    ("dining chair", "medium"):      (10, 35),
    ("dining chair", "large"):       (15, 50),
    ("office chair", "small"):       (10, 25),
    ("office chair", "medium"):      (15, 35),
    ("office chair", "large"):       (20, 50),
    ("armchair", "small"):           (20, 60),
    ("armchair", "medium"):          (30, 100),
    ("armchair", "large"):           (40, 150),
    ("recliner", "small"):           (25, 80),
    ("recliner", "medium"):          (30, 120),
    ("recliner", "large"):           (40, 150),
    ("rocking chair", "small"):      (10, 25),
    ("rocking chair", "medium"):     (15, 40),
    ("rocking chair", "large"):      (20, 55),
    ("bar stool", "small"):          (5, 10),
    ("bar stool", "medium"):         (8, 20),
    ("bar stool", "large"):          (12, 30),
    ("bench", "small"):              (10, 30),
    ("bench", "medium"):             (20, 60),
    ("bench", "large"):              (30, 100),
    ("ottoman", "small"):            (5, 15),
    ("ottoman", "medium"):           (10, 30),
    ("ottoman", "large"):            (15, 50),
    ("folding chair", "small"):      (2, 8),
    ("folding chair", "medium"):     (3, 12),
    ("folding chair", "large"):      (4, 15),
    ("desk chair", "small"):         (10, 25),
    ("desk chair", "medium"):        (15, 35),
    ("desk chair", "large"):         (20, 50),
    # ── Sofas ──
    ("sofa", "small"):               (40, 150),
    ("sofa", "medium"):              (65, 250),
    ("sofa", "large"):               (80, 350),
    ("loveseat", "small"):           (30, 100),
    ("loveseat", "medium"):          (45, 175),
    ("loveseat", "large"):           (55, 225),
    ("sectional sofa", "small"):     (60, 250),
    ("sectional sofa", "medium"):    (110, 450),
    ("sectional sofa", "large"):     (150, 600),
    ("futon", "small"):              (30, 100),
    ("futon", "medium"):             (45, 150),
    ("futon", "large"):              (55, 200),
    # ── Tables ──
    ("dining table", "small"):       (15, 50),
    ("dining table", "medium"):      (30, 120),
    ("dining table", "large"):       (50, 200),
    ("coffee table", "small"):       (8, 25),
    ("coffee table", "medium"):      (15, 50),
    ("coffee table", "large"):       (20, 75),
    ("end table", "small"):          (3, 10),
    ("end table", "medium"):         (5, 20),
    ("end table", "large"):          (8, 35),
    ("desk", "small"):               (15, 50),
    ("desk", "medium"):              (30, 100),
    ("desk", "large"):               (50, 175),
    ("nightstand", "small"):         (3, 15),
    ("nightstand", "medium"):        (5, 25),
    ("nightstand", "large"):         (8, 40),
    ("kitchen table", "small"):      (15, 50),
    ("kitchen table", "medium"):     (25, 100),
    ("kitchen table", "large"):      (40, 150),
    ("table", "small"):              (10, 35),
    ("table", "medium"):             (20, 70),
    ("table", "large"):              (35, 125),
    ("folding table", "small"):      (5, 15),
    ("folding table", "medium"):     (8, 25),
    ("folding table", "large"):      (12, 40),
    ("console table", "small"):      (5, 20),
    ("console table", "medium"):     (10, 40),
    ("console table", "large"):      (15, 60),
    ("square table", "small"):       (8, 30),
    ("square table", "medium"):      (15, 60),
    ("square table", "large"):       (25, 100),
    ("round table", "small"):        (8, 30),
    ("round table", "medium"):       (15, 60),
    ("round table", "large"):        (25, 100),
    # ── Beds ──
    ("bed", "small"):                (40, 100),    # twin
    ("bed", "medium"):               (65, 200),    # queen
    ("bed", "large"):                (80, 250),    # king
    ("queen bed", "medium"):         (65, 200),
    ("king bed", "large"):           (80, 250),
    ("twin bed", "small"):           (40, 100),
    ("bunk bed", "small"):           (40, 150),
    ("bunk bed", "medium"):          (50, 200),
    ("bunk bed", "large"):           (65, 275),
    ("mattress", "small"):           (25, 40),
    ("mattress", "medium"):          (35, 60),
    ("mattress", "large"):           (45, 80),
    ("box spring", "small"):         (20, 40),
    ("box spring", "medium"):        (30, 55),
    ("box spring", "large"):         (40, 70),
    ("crib", "small"):               (20, 40),
    ("crib", "medium"):              (25, 55),
    ("crib", "large"):               (30, 70),
    # ── Storage / Dressers ──
    ("dresser", "small"):            (20, 80),
    ("dresser", "medium"):           (40, 150),
    ("dresser", "large"):            (50, 250),
    ("chest of drawers", "small"):   (15, 60),
    ("chest of drawers", "medium"):  (25, 100),
    ("chest of drawers", "large"):   (35, 150),
    ("wardrobe", "small"):           (25, 80),
    ("wardrobe", "medium"):          (40, 140),
    ("wardrobe", "large"):           (60, 225),
    ("bookcase", "small"):           (10, 30),
    ("bookcase", "medium"):          (20, 70),
    ("bookcase", "large"):           (35, 120),
    ("filing cabinet", "small"):     (5, 30),
    ("filing cabinet", "medium"):    (10, 60),
    ("filing cabinet", "large"):     (15, 100),
    ("file cabinet", "small"):       (5, 30),
    ("file cabinet", "medium"):      (10, 60),
    ("file cabinet", "large"):       (15, 100),
    ("shelf", "small"):              (5, 15),
    ("shelf", "medium"):             (10, 30),
    ("shelf", "large"):              (20, 60),
    ("cabinet", "small"):            (10, 40),
    ("cabinet", "medium"):           (20, 80),
    ("cabinet", "large"):            (35, 150),
    ("tv stand", "small"):           (8, 30),
    ("tv stand", "medium"):          (15, 60),
    ("tv stand", "large"):           (25, 100),
    ("hall tree", "small"):          (10, 40),
    ("hall tree", "medium"):         (25, 80),
    ("hall tree", "large"):          (35, 125),
    # ── Appliances ──
    ("refrigerator", "small"):       (25, 150),
    ("refrigerator", "medium"):      (45, 250),
    ("refrigerator", "large"):       (55, 350),
    ("microwave", "small"):          (2, 15),
    ("microwave", "medium"):         (3, 30),
    ("microwave", "large"):          (5, 50),
    ("washing machine", "small"):    (25, 125),
    ("washing machine", "medium"):   (30, 175),
    ("washing machine", "large"):    (35, 225),
    ("dryer", "small"):              (25, 100),
    ("dryer", "medium"):             (30, 150),
    ("dryer", "large"):              (35, 200),
    ("dishwasher", "small"):         (20, 80),
    ("dishwasher", "medium"):        (25, 125),
    ("dishwasher", "large"):         (30, 150),
    ("chest freezer", "small"):      (10, 50),
    ("chest freezer", "medium"):     (20, 100),
    ("chest freezer", "large"):      (30, 150),
    ("stove", "small"):              (20, 100),
    ("stove", "medium"):             (30, 150),
    ("stove", "large"):              (35, 200),
    ("oven", "small"):               (20, 100),
    ("oven", "medium"):              (30, 150),
    ("oven", "large"):               (35, 200),
    ("coffee machine", "small"):     (1, 5),
    ("coffee machine", "medium"):    (2, 15),
    ("coffee machine", "large"):     (3, 25),
    ("toaster oven", "small"):       (1, 5),
    ("toaster oven", "medium"):      (2, 10),
    ("toaster oven", "large"):       (3, 15),
    ("water dispenser", "small"):    (5, 20),
    ("water dispenser", "medium"):   (8, 35),
    ("water dispenser", "large"):    (12, 50),
    ("water cooler", "small"):       (5, 20),
    ("water cooler", "medium"):      (8, 35),
    ("water cooler", "large"):       (12, 50),
    ("vending machine", "small"):    (30, 300),
    ("vending machine", "medium"):   (45, 500),
    ("vending machine", "large"):    (60, 700),
    # ── Electronics ──
    ("television", "small"):         (3, 15),
    ("television", "medium"):        (5, 30),
    ("television", "large"):         (8, 50),
    ("tv", "small"):                 (3, 15),
    ("tv", "medium"):                (5, 30),
    ("tv", "large"):                 (8, 50),
    ("computer monitor", "small"):   (2, 8),
    ("computer monitor", "medium"):  (4, 15),
    ("computer monitor", "large"):   (6, 25),
    ("monitor", "small"):            (2, 8),
    ("monitor", "medium"):           (4, 15),
    ("monitor", "large"):            (6, 25),
    ("laptop", "small"):             (0.5, 3),
    ("laptop", "medium"):            (0.5, 5),
    ("laptop", "large"):             (1, 8),
    ("printer", "small"):            (2, 10),
    ("printer", "medium"):           (4, 25),
    ("printer", "large"):            (6, 50),
    ("soundbar", "small"):           (1, 5),
    ("soundbar", "medium"):          (2, 10),
    ("soundbar", "large"):           (3, 15),
    # ── Outdoor / Garden ──
    ("barbecue grill", "small"):     (15, 50),
    ("barbecue grill", "medium"):    (30, 100),
    ("barbecue grill", "large"):     (50, 200),
    ("fire pit", "small"):           (10, 30),
    ("fire pit", "medium"):          (20, 70),
    ("fire pit", "large"):           (30, 120),
    ("outdoor chair", "small"):      (10, 20),
    ("outdoor chair", "medium"):     (15, 30),
    ("outdoor chair", "large"):      (20, 45),
    ("patio table", "small"):        (10, 30),
    ("patio table", "medium"):       (20, 60),
    ("patio table", "large"):        (30, 100),
    ("potted plant", "small"):       (2, 10),
    ("potted plant", "medium"):      (5, 25),
    ("potted plant", "large"):       (10, 50),
    ("planter", "small"):            (2, 10),
    ("planter", "medium"):           (5, 25),
    ("planter", "large"):            (10, 50),
    # ── Fitness ──
    ("treadmill", "small"):          (30, 150),
    ("treadmill", "medium"):         (40, 225),
    ("treadmill", "large"):          (50, 300),
    ("elliptical", "small"):         (25, 100),
    ("elliptical", "medium"):        (35, 175),
    ("elliptical", "large"):         (45, 250),
    ("weight bench", "small"):       (10, 40),
    ("weight bench", "medium"):      (15, 60),
    ("weight bench", "large"):       (20, 100),
    ("exercise bike", "small"):      (15, 40),
    ("exercise bike", "medium"):     (20, 70),
    ("exercise bike", "large"):      (25, 100),
    # ── Kids / Play ──
    ("jungle gym", "small"):         (10, 50),
    ("jungle gym", "medium"):        (15, 80),
    ("jungle gym", "large"):         (25, 150),
    ("trampoline", "small"):         (10, 30),
    ("trampoline", "medium"):        (20, 60),
    ("trampoline", "large"):         (30, 100),
    ("easel", "small"):              (3, 10),
    ("easel", "medium"):             (5, 20),
    ("easel", "large"):              (8, 35),
    ("stroller", "small"):           (5, 15),
    ("stroller", "medium"):          (8, 25),
    ("stroller", "large"):           (12, 35),
    # ── Misc Furniture ──
    ("mirror", "small"):             (2, 5),
    ("mirror", "medium"):            (5, 15),
    ("mirror", "large"):             (10, 30),
    ("lamp", "small"):               (1, 3),
    ("lamp", "medium"):              (3, 8),
    ("lamp", "large"):               (5, 15),
    ("rug", "small"):                (3, 10),
    ("rug", "medium"):               (8, 30),
    ("rug", "large"):                (15, 60),
    ("painting", "small"):           (2, 3),
    ("painting", "medium"):          (5, 8),
    ("painting", "large"):           (10, 15),
    ("coat rack", "small"):          (3, 8),
    ("coat rack", "medium"):         (5, 15),
    ("coat rack", "large"):          (8, 25),
    # ── Storage / Containers ──
    ("suitcase", "small"):           (3, 8),
    ("suitcase", "medium"):          (5, 12),
    ("suitcase", "large"):           (8, 18),
    ("plastic bin", "small"):        (3, 5),
    ("plastic bin", "medium"):       (5, 10),
    ("plastic bin", "large"):        (8, 15),
    ("toolbox", "small"):            (2, 10),
    ("toolbox", "medium"):           (5, 30),
    ("toolbox", "large"):            (10, 60),
    ("tool chest", "small"):         (15, 75),
    ("tool chest", "medium"):        (35, 175),
    ("tool chest", "large"):         (50, 300),
    ("safe", "small"):               (5, 50),
    ("safe", "medium"):              (10, 125),
    ("safe", "large"):               (25, 300),
    ("gun safe", "small"):           (10, 100),
    ("gun safe", "medium"):          (25, 200),
    ("gun safe", "large"):           (35, 400),
    # ── Office ──
    ("cubicle partition", "small"):  (5, 15),
    ("cubicle partition", "medium"): (10, 30),
    ("cubicle partition", "large"):  (15, 50),
    ("whiteboard", "small"):         (2, 5),
    ("whiteboard", "medium"):        (5, 15),
    ("whiteboard", "large"):         (8, 25),
    # ── Kitchen Small Items ──
    ("dish rack", "small"):          (1, 3),
    ("dish rack", "medium"):         (2, 5),
    ("dish rack", "large"):          (3, 8),
    ("blender", "small"):            (0.5, 3),
    ("blender", "medium"):           (1, 5),
    ("blender", "large"):            (1.5, 8),
    ("food container", "small"):     (0.5, 1),
    ("food container", "medium"):    (1, 2),
    ("food container", "large"):     (2, 4),
    # ── Safety ──
    ("fire extinguisher", "small"):  (1, 5),
    ("fire extinguisher", "medium"): (1.5, 10),
    ("fire extinguisher", "large"):  (2, 15),
    # ── Miscellaneous ──
    ("trash can", "small"):          (2, 5),
    ("trash can", "medium"):         (3, 8),
    ("trash can", "large"):          (5, 15),
    ("backpack", "small"):           (1, 3),
    ("backpack", "medium"):          (2, 5),
    ("backpack", "large"):           (3, 8),
    ("hard hat", "small"):           (0.5, 1),
    ("hard hat", "medium"):          (0.5, 1),
    ("hard hat", "large"):           (0.5, 2),
    ("helmet", "small"):             (0.5, 1),
    ("helmet", "medium"):            (0.5, 2),
    ("helmet", "large"):             (1, 3),
    ("vacuum cleaner", "small"):     (3, 10),
    ("vacuum cleaner", "medium"):    (5, 18),
    ("vacuum cleaner", "large"):     (8, 25),
    ("water bottle", "small"):       (0.5, 2),
    ("water bottle", "medium"):      (1, 5),
    ("water bottle", "large"):       (2, 10),
    ("bowl", "small"):               (0.2, 0.5),
    ("bowl", "medium"):              (0.5, 1),
    ("bowl", "large"):               (1, 2),
    ("decorative screen", "small"):  (3, 8),
    ("decorative screen", "medium"): (5, 15),
    ("decorative screen", "large"):  (8, 25),
}

# Weight-per-cuft fallback when we only have dimensions
_DEFAULT_DENSITY_LBS_PER_CUFT = 7.0  # typical household goods


def _normalize_name(name: str) -> str:
    """Normalize item name for lookup: lowercase, strip, remove plurals."""
    name = name.strip().lower()
    # Common aliases
    aliases = {
        "chair arm regular": "armchair",
        "chair rocking": "rocking chair",
        "tv stand wood": "tv stand",
        "painting/picture large": "painting",
        "computer": "laptop",
        "night stand": "nightstand",
        "file cabinet": "filing cabinet",
        "cubicle wall": "cubicle partition",
    }
    if name in aliases:
        return aliases[name]
    return name


def lookup_volume_weight(
    name: str,
    size: str = "medium",
    dimensions: dict | None = None,
    count: int = 1,
) -> tuple[float, float]:
    """Look up volume (cu ft) and weight (lbs) for a single unit of an item.

    Priority:
    1. Exact (name, size) match in lookup table
    2. (name, "medium") fallback
    3. Compute from LLM-estimated dimensions
    4. Default estimate based on size category

    Returns:
        (volume_cuft, weight_lbs) for ONE item (not multiplied by count)
    """
    norm = _normalize_name(name)
    size = (size or "medium").strip().lower()
    if size not in ("small", "medium", "large"):
        size = "medium"

    # 1. Exact match
    key = (norm, size)
    if key in _VOLUME_TABLE:
        return _VOLUME_TABLE[key]

    # 2. Fallback to medium
    med_key = (norm, "medium")
    if med_key in _VOLUME_TABLE:
        vol, wt = _VOLUME_TABLE[med_key]
        scale = {"small": 0.6, "medium": 1.0, "large": 1.5}.get(size, 1.0)
        return round(vol * scale, 1), round(wt * scale, 1)

    # 3. Partial name match — try matching last word or substring
    for table_name, table_size in _VOLUME_TABLE:
        if table_size == size and (
            norm.endswith(table_name) or table_name.endswith(norm)
            or table_name in norm or norm in table_name
        ):
            return _VOLUME_TABLE[(table_name, table_size)]

    # 4. Compute from dimensions
    if dimensions and all(dimensions.get(k, 0) > 0 for k in ("length_in", "width_in", "height_in")):
        vol_cubic_in = (
            dimensions["length_in"] * dimensions["width_in"] * dimensions["height_in"]
        )
        vol_cuft = round(vol_cubic_in / 1728, 1)  # 12^3
        weight = round(vol_cuft * _DEFAULT_DENSITY_LBS_PER_CUFT, 1)
        return vol_cuft, weight

    # 5. Default by size category
    defaults = {
        "small": (5, 20),
        "medium": (15, 60),
        "large": (35, 150),
    }
    return defaults.get(size, (15, 60))


def estimate_packing_materials(items: list[dict]) -> dict:
    """Estimate packing materials needed based on inventory.

    Args:
        items: list of dicts with 'name', 'count', 'room', 'notes'

    Returns:
        dict with material types and quantities
    """
    total_items = sum(item.get("count", 1) for item in items)
    rooms = set(item.get("room", "") for item in items)
    num_rooms = max(len(rooms), 1)

    # Heuristics based on industry standards
    has_kitchen = any("kitchen" in (item.get("room") or "").lower() for item in items)
    has_bedroom = any("bedroom" in (item.get("room") or "").lower() for item in items)
    has_closet = any("closet" in (item.get("room") or "").lower() for item in items)
    has_fragile = any(
        any(f in (item.get("name") or "").lower() for f in
            ("tv", "television", "monitor", "lamp", "mirror", "glass"))
        for item in items
    )

    materials = {
        "small_boxes": max(5, num_rooms * 3),
        "medium_boxes": max(5, num_rooms * 5),
        "large_boxes": max(3, num_rooms * 3),
        "dish_pack_boxes": 2 if has_kitchen else 0,
        "wardrobe_boxes": 3 if has_bedroom or has_closet else 0,
        "picture_boxes": 2 if has_fragile else 0,
        "packing_paper_bundles": max(1, 1 + (1 if has_kitchen else 0)),
        "plastic_wrap_rolls": max(1, total_items // 20),
        "mattress_bags": sum(
            item.get("count", 1) for item in items
            if any(w in (item.get("name") or "").lower() for w in ("bed", "mattress"))
        ),
    }
    return materials


def estimate_truck_size(total_volume_cuft: float) -> str:
    """Recommend truck size based on total volume."""
    if total_volume_cuft <= 200:
        return "Cargo van or 10-foot truck"
    elif total_volume_cuft <= 500:
        return "15-foot truck"
    elif total_volume_cuft <= 800:
        return "20-foot truck"
    elif total_volume_cuft <= 1200:
        return "26-foot truck"
    else:
        return "26-foot truck + additional vehicle"
