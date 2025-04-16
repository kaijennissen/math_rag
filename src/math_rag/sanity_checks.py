import logging
from math_rag.atomic_unit import AtomicUnit
import coloredlogs
import json
from math_rag.project_root import ROOT

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="WARNING",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

ATOMIC_UNITS_PATH = ROOT / "docs/atomic_units"

if __name__ == "__main__":
    for json_file in ATOMIC_UNITS_PATH.glob("subsection_*_*_units.json"):
        logger.info(f"  Processing file: {json_file.name}")
        with open(json_file, "r") as f:
            data = json.load(f)
        atomic_units = [AtomicUnit.from_dict(unit) for unit in data.get("chunks")]
