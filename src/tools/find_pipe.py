"""Tool for finding pipes in P&ID diagrams."""

from typing import Dict, Any, Optional
from src.utils.logger import setup_logger
from src.ocr.paddle_ocr import PaddleOCRExtractor

logger = setup_logger(__name__)


def find_pipe(
    image_path: str,
    pipe_label: str,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Find a specific pipe in a P&ID diagram.

    Args:
        image_path: Path to the P&ID image
        pipe_label: Label of the pipe to find (e.g., "P-101")
        confidence_threshold: Minimum OCR confidence

    Returns:
        Dictionary with pipe location and metadata
    """
    try:
        ocr_extractor = PaddleOCRExtractor()
        regions = ocr_extractor.extract_text_with_regions(
            image_path, confidence_threshold
        )

        # Find the pipe
        for region in regions:
            if region["text"].strip().upper() == pipe_label.strip().upper():
                logger.info(f"Found pipe: {pipe_label}")
                return {
                    "found": True,
                    "pipe_label": pipe_label,
                    "location": region["bbox"],
                    "confidence": region["confidence"],
                }

        logger.warning(f"Pipe not found: {pipe_label}")
        return {
            "found": False,
            "pipe_label": pipe_label,
            "error": f"Pipe {pipe_label} not found in image",
        }
    except Exception as e:
        logger.error(f"Error finding pipe: {e}")
        return {
            "found": False,
            "pipe_label": pipe_label,
            "error": str(e),
        }
