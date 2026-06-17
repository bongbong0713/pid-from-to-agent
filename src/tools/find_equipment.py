"""Tool for finding equipment in P&ID diagrams."""

from typing import Dict, Any, List
from src.utils.logger import setup_logger
from src.vision.equipment_detector import EquipmentDetector

logger = setup_logger(__name__)


def find_equipment(
    image_path: str,
    equipment_label: str = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Find equipment in a P&ID diagram.

    Args:
        image_path: Path to the P&ID image
        equipment_label: Specific equipment to find (optional)
        confidence_threshold: Minimum OCR confidence

    Returns:
        Dictionary with equipment locations and metadata
    """
    try:
        detector = EquipmentDetector()
        equipment_list = detector.detect_equipment(
            image_path, confidence_threshold
        )

        if equipment_label:
            # Find specific equipment
            eq = detector.find_equipment_by_label(equipment_list, equipment_label)
            if eq:
                logger.info(f"Found equipment: {equipment_label}")
                return {
                    "found": True,
                    "equipment_label": equipment_label,
                    "equipment": eq,
                }
            else:
                logger.warning(f"Equipment not found: {equipment_label}")
                return {
                    "found": False,
                    "equipment_label": equipment_label,
                    "error": f"Equipment {equipment_label} not found",
                }
        else:
            # Return all equipment
            logger.info(f"Found {len(equipment_list)} equipment items")
            return {
                "found": True,
                "equipment_list": equipment_list,
                "count": len(equipment_list),
            }
    except Exception as e:
        logger.error(f"Error finding equipment: {e}")
        return {
            "found": False,
            "error": str(e),
        }
