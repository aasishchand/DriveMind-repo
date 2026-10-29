"""Canonical model labels. 0 is background; foreground IDs start at 1."""

CLASS_NAMES = ("ambulance", "fire_truck", "police_vehicle", "ordinary_vehicle")
CLASS_TO_ID = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}


def normalize_label(label: str) -> str:
    key = "_".join(label.strip().lower().replace("-", " ").split())
    aliases = {"firetruck": "fire_truck", "fire_engine": "fire_truck",
               "police": "police_vehicle", "police_car": "police_vehicle",
               "car": "ordinary_vehicle", "truck": "ordinary_vehicle",
               "vehicle": "ordinary_vehicle", "ordinary": "ordinary_vehicle"}
    key = aliases.get(key, key)
    if key not in CLASS_TO_ID:
        raise ValueError(f"Unknown emergency detector class {label!r}; expected one of {CLASS_NAMES}")
    return key
