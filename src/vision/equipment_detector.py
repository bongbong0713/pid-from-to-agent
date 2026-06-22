"""Equipment detection module."""

import cv2
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import setup_logger
from src.ocr.paddle_ocr import PaddleOCRExtractor

logger = setup_logger(__name__)


class EquipmentDetector:
    """Detect equipment labels and types in P&ID diagrams."""

    # Standard P&ID equipment codes
    EQUIPMENT_PATTERNS = {
        "Pump": [r"^P-\d+"],
        "Tank": [r"^T-\d+"],
        "Reactor": [r"^E-\d+"],
        "Compressor": [r"^C-\d+"],
        "Cooler": [r"^E-\d+C"],
        "Heater": [r"^E-\d+H"],
        "Valve": [r"^V-\d+"],
        "Filter": [r"^F-\d+"],
        "Separator": [r"^S-\d+"],
        "Accumulator": [r"^AC-\d+"],
    }

    def __init__(self, language: str = "en"):
        """
        Initialize equipment detector.

        Args:
            language: Language for OCR
        """
        self.ocr_extractor = PaddleOCRExtractor(language=language)
        logger.info("Initialized EquipmentDetector")

    def detect_equipment(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect equipment in a P&ID image.

        Args:
            image_path: Path to the image
            confidence_threshold: Minimum OCR confidence

        Returns:
            List of detected equipment with metadata
        """
        # Extract text regions
        regions = self.ocr_extractor.extract_text_with_regions(
            image_path, confidence_threshold
        )

        # Classify equipment and normalize labels
        equipment = []
        import re
        for region in regions:
            raw_text = region.get("text", "")
            label = raw_text.strip().upper()

            # Repair common OCR issues: leading dash or digits-only (e.g., "-3118" -> "E-3118")
            if re.match(r"^-\d+$", label):
                label = "E" + label
            elif re.match(r"^\d{3,}$", label):
                label = "E-" + label

            # Update the region text to normalized label so GraphBuilder stores normalized labels
            region["text"] = label

            equipment_type = self._classify_equipment(label)
            if equipment_type:
                region["type"] = equipment_type
                equipment.append(region)

        logger.info(f"Detected {len(equipment)} equipment items")
        return equipment

    def _classify_equipment(self, label: str) -> str:
        """
        Classify equipment based on label.

        Args:
            label: Equipment label

        Returns:
            Equipment type or None
        """
        import re

        label = label.strip().upper()

        # Repair common OCR issues: leading dash or digits-only (e.g., "-3118" -> "E-3118")
        import re
        if re.match(r"^-\d+$", label):
            label = "E" + label
        elif re.match(r"^\d{3,}$", label):
            label = "E-" + label

        for equipment_type, patterns in self.EQUIPMENT_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, label):
                    return equipment_type

        return None

    def find_equipment_by_label(
        self,
        equipment_list: List[Dict[str, Any]],
        label: str,
    ) -> Dict[str, Any]:
        """
        Find equipment by label.

        Args:
            equipment_list: List of detected equipment
            label: Equipment label to find

        Returns:
            Equipment dictionary or None
        """
        label = label.strip().upper()
        for eq in equipment_list:
            if eq["text"].strip().upper() == label:
                return eq
        return None

    def find_equipment_near_point(
        self,
        equipment_list: List[Dict[str, Any]],
        point: tuple,
        distance_threshold: float = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find equipment near a specific point.

        Args:
            equipment_list: List of detected equipment
            point: (x, y) coordinate
            distance_threshold: Maximum distance in pixels

        Returns:
            List of nearby equipment
        """
        x_p, y_p = point
        nearby = []

        for eq in equipment_list:
            center_x = eq["bbox"]["center_x"]
            center_y = eq["bbox"]["center_y"]

            distance = np.sqrt((x_p - center_x) ** 2 + (y_p - center_y) ** 2)
            if distance <= distance_threshold:
                eq["distance"] = float(distance)
                nearby.append(eq)

        # Sort by distance
        nearby.sort(key=lambda x: x["distance"])
        return nearby

    def visualize_equipment(
        self,
        image_path: str,
        equipment_list: List[Dict[str, Any]],
        output_path: str = None,
    ) -> np.ndarray:
        """
        Visualize detected equipment.

        Args:
            image_path: Path to the image
            equipment_list: List of detected equipment
            output_path: Path to save visualization (optional)

        Returns:
            Visualization image
        """
        image = cv2.imread(image_path)

        # Draw equipment bounding boxes
        for eq in equipment_list:
            bbox = eq["bbox"]
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            x_max = int(bbox["x_max"])
            y_max = int(bbox["y_max"])

            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw label
            label = f"{eq['text']} ({eq['type']})"
            cv2.putText(
                image,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Saved equipment visualization to: {output_path}")

        return image
