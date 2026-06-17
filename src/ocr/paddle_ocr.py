"""PaddleOCR text extraction module."""

import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PaddleOCRExtractor:
    """Extract text and bounding boxes from images using PaddleOCR."""

    def __init__(self, language: str = "en", use_gpu: bool = False):
        """
        Initialize PaddleOCR extractor.

        Args:
            language: Language code (e.g., 'en', 'ch', 'ja')
            use_gpu: Whether to use GPU acceleration
        """
        logger.info(f"Initializing PaddleOCR with language: {language}")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=language,
            use_gpu=use_gpu,
        )
        self.language = language

    def extract_text(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """
        Extract text and bounding boxes from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - texts: List of detected text strings
                - boxes: List of bounding box coordinates
                - confidences: List of confidence scores
                - raw_results: Raw OCR output
        """
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Extracting text from: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            raise ValueError(f"Failed to read image: {image_path}")

        # Run OCR
        results = self.ocr.ocr(image, cls=True)

        # Parse results
        texts = []
        boxes = []
        confidences = []

        if results and results[0]:
            for line in results[0]:
                # line format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                bbox, (text, confidence) = line
                texts.append(text)
                boxes.append(bbox)
                confidences.append(confidence)

        logger.info(f"Extracted {len(texts)} text regions")

        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences,
            "raw_results": results,
            "image_shape": image.shape,
        }

    def extract_text_with_regions(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Extract text with regional information.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score to include

        Returns:
            List of dictionaries with text and metadata
        """
        ocr_result = self.extract_text(image_path)

        regions = []
        for text, box, confidence in zip(
            ocr_result["texts"],
            ocr_result["boxes"],
            ocr_result["confidences"],
        ):
            if confidence >= confidence_threshold:
                # Convert box format to min/max coordinates
                box_array = np.array(box)
                x_coords = box_array[:, 0]
                y_coords = box_array[:, 1]

                region = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x_min": float(x_coords.min()),
                        "y_min": float(y_coords.min()),
                        "x_max": float(x_coords.max()),
                        "y_max": float(y_coords.max()),
                        "center_x": float(x_coords.mean()),
                        "center_y": float(y_coords.mean()),
                    },
                    "raw_box": box,
                }
                regions.append(region)

        logger.info(f"Extracted {len(regions)} regions with confidence >= {confidence_threshold}")
        return regions

    def visualize_ocr(
        self,
        image_path: str,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Create visualization of OCR results.

        Args:
            image_path: Path to the input image
            output_path: Path to save the visualization (optional)

        Returns:
            Visualization image as numpy array
        """
        image = cv2.imread(image_path)
        ocr_result = self.extract_text(image_path)

        # Draw bounding boxes and text
        for text, box, confidence in zip(
            ocr_result["texts"],
            ocr_result["boxes"],
            ocr_result["confidences"],
        ):
            box_array = np.array(box, dtype=np.int32)
            cv2.polylines(image, [box_array], True, (0, 255, 0), 2)

            # Put text label
            x_min = int(box_array[:, 0].min())
            y_min = int(box_array[:, 1].min())
            label = f"{text} ({confidence:.2f})"
            cv2.putText(
                image,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Saved visualization to: {output_path}")

        return image
