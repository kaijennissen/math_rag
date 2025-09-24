import json
import logging

import coloredlogs

from math_rag.core import ROOT, AtomicItem

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="WARNING",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

ATOMIC_ITEMS_PATH = ROOT / "docs/atomic_units"

if __name__ == "__main__":
    for json_file in ATOMIC_ITEMS_PATH.glob("subsection_*_*_units.json"):
        logger.info(f"  Processing file: {json_file.name}")
        with open(json_file, "r") as f:
            data = json.load(f)
        atomic_units = [AtomicItem.from_dict(unit) for unit in data.get("chunks")]
