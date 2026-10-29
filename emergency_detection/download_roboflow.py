"""Download DriveMind's three Roboflow vehicle datasets in YOLOv8 format."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "datasets"
PROJECTS = {
    "general-vehicles": "vehicle-detection-by9xs-ttuog",
    "emergency-vehicles": "emergency-vehicle-szehs-kzbpw",
    "itawd": "itawd-yinz0-ykdth",
}
WORKSPACE = "aasishchand-narra"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="parent directory for the three YOLOv8 exports")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        parser.error("set ROBOFLOW_API_KEY in the environment; do not put the key in source code")

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("Roboflow is not installed. Run: python -m pip install roboflow") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(WORKSPACE)
    for folder, project_slug in PROJECTS.items():
        location = (args.output / folder).resolve()
        project = workspace.project(project_slug)
        dataset = project.version(1).download("yolov8", location=str(location))
        print(f"{folder}: {dataset.location}")
        print("  Review data.yaml classes before preparing the combined dataset.")


if __name__ == "__main__":
    main()
