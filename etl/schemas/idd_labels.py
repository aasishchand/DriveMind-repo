"""pandera schemas for India Driving Dataset (IDD) labels, flattened to rows.

Two schemas cover the four IDD variants:

    IDDDetectionLabelSchema      -> Detection + Temporal (one row per bounding box)
    IDDSegmentationLabelSchema   -> Segmentation + Lite  (one row per polygon/instance)

``IDD_LEVEL3_CLASSES`` and ``IDD_LEVEL3_ID_MAP`` are VERIFIED: the class names
were collected from a full scan of a real IDD Segmentation Part 2 train+val set
(6,982 ``*_gtFine_polygons.json`` files, 41 distinct labels; "ground" added as a
42nd, seen 3x in that scan but absent from the official table below), and the
id mapping is transcribed from the official devkit
(AutoNUE/public-code, ``helpers/anue_labels.py``, the ``level3Id`` field) -
multiple names legitimately share one id (e.g. "caravan"/"trailer"/"train"/
"vehicle fallback" -> 12; the five void/ignore labels -> 255).

``IDD_DETECTION_CLASSES`` is still a PLACEHOLDER - no real IDD Detection data
has gone through this pipeline yet. Reconcile it against the Pascal-VOC class
set shipped with IDD Detection before relying on it.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.20
    import pandera.pandas as pa
    from pandera.typing.pandas import Series
except ModuleNotFoundError:  # pandera < 0.20
    import pandera as pa
    from pandera.typing import Series


IDD_SPLITS = ("train", "val", "test")
IDD_VARIANTS = ("detection", "temporal", "segmentation", "lite")

# PLACEHOLDER - reconcile with the IDD Detection VOC class set.
IDD_DETECTION_CLASSES = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "autorickshaw",
    "person",
    "rider",
    "animal",
    "traffic sign",
    "traffic light",
    "vehicle fallback",
    "caravan",
    "trailer",
    "train",
)

# All 42 label names actually observed (see module docstring), ordered by
# level3Id then alphabetically within each id group.
IDD_LEVEL3_CLASSES = (
    "road",                    # 0
    "drivable fallback",       # 1
    "parking",                 # 1
    "sidewalk",                # 2
    "non-drivable fallback",   # 3
    "rail track",              # 3
    "animal",                  # 4
    "person",                  # 4
    "rider",                   # 5
    "motorcycle",              # 6
    "bicycle",                 # 7
    "autorickshaw",            # 8
    "car",                     # 9
    "truck",                   # 10
    "bus",                     # 11
    "caravan",                 # 12
    "train",                   # 12
    "trailer",                 # 12
    "vehicle fallback",        # 12
    "curb",                    # 13
    "wall",                    # 14
    "fence",                   # 15
    "guard rail",              # 16
    "billboard",               # 17
    "traffic sign",            # 18
    "traffic light",           # 19
    "pole",                    # 20
    "polegroup",               # 20
    "obs-str-bar-fallback",    # 21
    "building",                # 22
    "bridge",                  # 23
    "tunnel",                  # 23
    "vegetation",              # 24
    "fallback background",     # 25
    "sky",                     # 25
    # void / ignore group (level3Id 255)
    "ego vehicle",
    "ground",                  # not in the official table - defaulted to void, see map below
    "license plate",
    "out of roi",
    "rectification border",
    "unlabeled",
)

# Official name -> level3Id, transcribed from AutoNUE/public-code
# helpers/anue_labels.py. "ground" is not in the official table; it showed up
# 3 times in the real scan and is mapped to 255 (void/ignore) as the safest
# default - flag it if a real use turns out to need it treated differently.
IDD_LEVEL3_ID_MAP: dict[str, int] = {
    "road": 0,
    "parking": 1,
    "drivable fallback": 1,
    "sidewalk": 2,
    "rail track": 3,
    "non-drivable fallback": 3,
    "person": 4,
    "animal": 4,
    "rider": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "autorickshaw": 8,
    "car": 9,
    "truck": 10,
    "bus": 11,
    "caravan": 12,
    "trailer": 12,
    "train": 12,
    "vehicle fallback": 12,
    "curb": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "billboard": 17,
    "traffic sign": 18,
    "traffic light": 19,
    "pole": 20,
    "polegroup": 20,
    "obs-str-bar-fallback": 21,
    "building": 22,
    "bridge": 23,
    "tunnel": 23,
    "vegetation": 24,
    "sky": 25,
    "fallback background": 25,
    "unlabeled": 255,
    "ego vehicle": 255,
    "rectification border": 255,
    "out of roi": 255,
    "license plate": 255,
    "ground": 255,
}


class IDDDetectionLabelSchema(pa.DataFrameModel):
    """One row per annotated bounding box (Detection and Temporal variants)."""

    image_id: Series[str] = pa.Field(nullable=False, description="Unique image identifier.")
    image_path: Series[str] = pa.Field(
        nullable=False, description="Image path relative to the Bronze IDD root."
    )
    image_width: Series[int] = pa.Field(gt=0, description="Image width in pixels.")
    image_height: Series[int] = pa.Field(gt=0, description="Image height in pixels.")
    split: Series[str] = pa.Field(isin=IDD_SPLITS, description="Dataset split.")
    variant: Series[str] = pa.Field(
        isin=("detection", "temporal"), description="Source IDD variant."
    )
    sequence_id: Series[str] = pa.Field(
        nullable=True, description="Video sequence id (Temporal only; null for Detection)."
    )
    frame_index: Series[pd.Int64Dtype] = pa.Field(
        ge=0, nullable=True, description="Frame position within the sequence (Temporal only)."
    )
    class_name: Series[str] = pa.Field(
        isin=IDD_DETECTION_CLASSES, description="Canonical object class."
    )
    class_id: Series[pd.Int64Dtype] = pa.Field(
        ge=0, description="Integer class id (index into IDD_DETECTION_CLASSES)."
    )
    bbox_x: Series[float] = pa.Field(ge=0.0, description="Box top-left x in pixels.")
    bbox_y: Series[float] = pa.Field(ge=0.0, description="Box top-left y in pixels.")
    bbox_w: Series[float] = pa.Field(gt=0.0, description="Box width in pixels.")
    bbox_h: Series[float] = pa.Field(gt=0.0, description="Box height in pixels.")
    occluded: Series[bool] = pa.Field(nullable=True, description="Annotator 'occluded' flag.")
    truncated: Series[bool] = pa.Field(nullable=True, description="Annotator 'truncated' flag.")

    class Config:
        name = "idd_detection_labels_silver"
        strict = True
        coerce = True
        ordered = False

    @pa.dataframe_check(name="bbox_within_image_bounds")
    def _bbox_within_image(cls, df: pd.DataFrame) -> "pd.Series":
        # +1 px slack for rounding at the far edge.
        return (df["bbox_x"] + df["bbox_w"] <= df["image_width"] + 1) & (
            df["bbox_y"] + df["bbox_h"] <= df["image_height"] + 1
        )


class IDDSegmentationLabelSchema(pa.DataFrameModel):
    """One row per polygon / instance (Segmentation and Lite variants)."""

    image_id: Series[str] = pa.Field(nullable=False, description="Unique image identifier.")
    image_path: Series[str] = pa.Field(
        nullable=False, description="Image path relative to the Bronze IDD root."
    )
    image_width: Series[int] = pa.Field(gt=0, description="Image width in pixels.")
    image_height: Series[int] = pa.Field(gt=0, description="Image height in pixels.")
    split: Series[str] = pa.Field(isin=IDD_SPLITS, description="Dataset split.")
    variant: Series[str] = pa.Field(
        isin=("segmentation", "lite"), description="Source IDD variant."
    )
    gt_json_path: Series[str] = pa.Field(
        nullable=False, description="Path to the source Cityscapes-style polygon JSON."
    )
    label: Series[str] = pa.Field(
        isin=IDD_LEVEL3_CLASSES, description="Canonical level-3 class name."
    )
    level3_id: Series[pd.Int64Dtype] = pa.Field(
        ge=0,
        le=255,
        description=(
            "Official AutoNUE level3Id (see IDD_LEVEL3_ID_MAP). "
            "255 = void/ignore (unlabeled, ego vehicle, out of roi, ...)."
        ),
    )
    instance_id: Series[pd.Int64Dtype] = pa.Field(
        ge=0, nullable=True, description="Instance id within the image (null for stuff classes)."
    )
    polygon_area_px: Series[float] = pa.Field(
        ge=0.0, nullable=True, description="Polygon area in pixels."
    )
    num_vertices: Series[pd.Int64Dtype] = pa.Field(
        ge=3, nullable=True, description="Vertex count of the polygon."
    )

    class Config:
        name = "idd_segmentation_labels_silver"
        strict = True
        coerce = True
        ordered = False


def validate_detection_labels(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    return IDDDetectionLabelSchema.validate(df, lazy=lazy)


def validate_segmentation_labels(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    return IDDSegmentationLabelSchema.validate(df, lazy=lazy)
