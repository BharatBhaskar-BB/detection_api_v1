"""150-item household checklist for LLM priming."""

CHECKLIST: list[str] = [
    # ── Living Room / Lounge ──
    "sofa", "loveseat", "sectional sofa", "armchair", "recliner", "ottoman",
    "bean bag", "futon", "coffee table", "side table", "end table",
    "console table", "TV", "TV stand", "media console", "entertainment center",
    "floor lamp", "table lamp", "desk lamp", "chandelier", "pendant light",
    "wall sconce", "ceiling fan", "bookshelf", "bookcase", "shelving unit",
    "display cabinet", "fireplace", "mantel",

    # ── Dining ──
    "dining table", "conference table", "dining chair", "bar stool", "bench", "buffet",
    "sideboard", "china cabinet", "wine rack",

    # ── Kitchen ──
    "refrigerator", "oven", "stove", "microwave", "dishwasher",
    "toaster", "toaster oven", "blender", "coffee maker", "kettle",
    "kitchen island", "kitchen counter", "kitchen cart", "pantry cabinet",
    "spice rack",

    # ── Bedroom ──
    "bed", "headboard", "nightstand", "bedside table", "dresser",
    "chest of drawers", "wardrobe", "armoire", "vanity", "vanity mirror",
    "jewelry box", "bed frame", "mattress",

    # ── Bathroom ──
    "towel rack", "towel bar", "bathroom cabinet", "medicine cabinet",
    "laundry basket", "hamper", "shower curtain", "bath mat",
    "soap dispenser", "toilet paper holder",

    # ── Office / Study ──
    "desk", "writing desk", "standing desk", "office chair", "ergonomic chair",
    "filing cabinet", "monitor", "computer", "laptop", "printer",
    "whiteboard", "bulletin board",

    # ── Laundry ──
    "washing machine", "dryer", "ironing board", "iron",
    "drying rack", "laundry shelf",

    # ── Decor ──
    "rug", "area rug", "carpet", "curtains", "blinds", "shutters",
    "painting", "wall art", "photo frame", "wall clock", "vase",
    "plant", "potted plant", "artificial plant", "candle", "candle holder",
    "mirror", "wall mirror", "decorative pillow", "throw blanket",

    # ── Storage / Utility ──
    "shoe rack", "coat rack", "hat stand", "umbrella stand",
    "magazine rack", "storage bin", "storage basket", "trunk",
    "toy chest", "toy box",

    # ── Fitness / Hobby ──
    "exercise bike", "treadmill", "elliptical", "yoga mat",
    "weight bench", "dumbbells", "stationary bike",
    "piano", "keyboard piano", "guitar stand", "record player",

    # ── Baby / Kids ──
    "baby crib", "high chair", "changing table", "playpen",
    "rocking chair", "nursing chair",

    # ── Pets ──
    "pet bed", "cat tree", "aquarium", "bird cage", "pet crate",

    # ── Garage / Outdoor ──
    "workbench", "tool chest", "lawn mower", "bicycle",
    "patio table", "patio chair", "grill", "bbq grill",
    "outdoor umbrella", "garden hose reel",

    # ── Misc ──
    "standing fan", "space heater", "humidifier", "dehumidifier",
    "air purifier", "vacuum cleaner", "robot vacuum",
    "safe", "mini fridge", "water cooler",
    "sewing machine", "paper shredder",

    # ── Personal / Everyday ──
    "backpack", "suitcase", "duffel bag", "briefcase",
    "helmet", "hat", "handbag", "tote bag",
    "umbrella", "walking stick", "skateboard",
    "luggage", "garment bag",
]

# Deduplicate and sort
CHECKLIST = sorted(set(CHECKLIST))


def get_checklist_text() -> str:
    """Format checklist as comma-separated string for LLM prompt."""
    return ", ".join(CHECKLIST)


def get_checklist_count() -> int:
    return len(CHECKLIST)
