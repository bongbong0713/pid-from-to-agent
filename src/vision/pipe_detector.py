"""Pipe detection module using OpenCV and regex pattern matching."""

import cv2
import numpy as np
import re
from typing import List, Tuple, Dict, Any
from src.utils.logger import setup_logger
from src.vision.preprocessing import ImagePreprocessor
from src.ocr.paddle_ocr import PaddleOCRExtractor

logger = setup_logger(__name__)


class PipeDetector:
    """Detect pipe line segments and labels in P&ID diagrams."""

    # Pipe label patterns (P&ID standard formats)
    PIPE_LABEL_PATTERNS = [
        r"\d{3}-P-\d{6}-[A-Z0-9]{2,}-[A-Z0-9]{2,}",  # e.g., 300-P-310305-NB01-HC
        r"P-\d+",  # Simple format e.g., P-101
        r"[A-Z]{2}-P-\d+",  # Format e.g., HC-P-101
    ]

    def __init__(
        self,
        canny_threshold1: float = 50,
        canny_threshold2: float = 150,
        hough_rho: float = 1.0,
        hough_theta: float = np.pi / 180,
        hough_threshold: int = 50,
        hough_min_length: float = 30,
        hough_max_gap: float = 10,
    ):
        """
        Initialize pipe detector.

        Args:
            canny_threshold1: Lower threshold for Canny edge detection
            canny_threshold2: Upper threshold for Canny edge detection
            hough_rho: Distance resolution in pixels
            hough_theta: Angle resolution in radians
            hough_threshold: Minimum number of votes
            hough_min_length: Minimum line length
            hough_max_gap: Maximum gap between line segments
        """
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.ocr_extractor = PaddleOCRExtractor()

    def detect_pipe_labels(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect pipe labels using OCR and regex patterns.

        Args:
            image_path: Path to the image
            confidence_threshold: Minimum OCR confidence

        Returns:
            List of detected pipe labels with metadata
        """
        # Extract all text regions
        regions = self.ocr_extractor.extract_text_with_regions(
            image_path, confidence_threshold
        )

        pipe_labels = []
        for region in regions:
            raw = region["text"].strip()
            # Normalize: remove extra spaces and uppercase
            norm_text = re.sub(r"\s+", "", raw).upper()
            region["text"] = norm_text
            # Check against pipe label patterns
            if self._is_pipe_label(norm_text):
                region["type"] = "pipe"
                pipe_labels.append(region)

        logger.info(f"Detected {len(pipe_labels)} pipe labels")
        return pipe_labels

    def _is_pipe_label(self, text: str) -> bool:
        """
        Check if text matches pipe label patterns.

        Args:
            text: Text to check

        Returns:
            True if text matches a pipe label pattern
        """
        text = text.strip().upper()

        # Try direct pattern match first
        for pattern in self.PIPE_LABEL_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Fallback: tolerate OCR errors like missing leading groups or spaces
        cleaned = re.sub(r"\s+", "", text)
        if re.search(r"P-\d+", cleaned):
            return True

        return False

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Edge map
        """
        edges = cv2.Canny(
            image,
            self.canny_threshold1,
            self.canny_threshold2,
        )
        logger.info("Detected edges using Canny edge detection")
        return edges

    def detect_lines(
        self,
        image: np.ndarray,
        use_hough_p: bool = True,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect line segments.

        Args:
            image: Preprocessed grayscale image
            use_hough_p: Whether to use HoughLinesP (True) or HoughLines (False)

        Returns:
            List of line segments as ((x1, y1), (x2, y2))
        """
        # Detect edges
        edges = self.detect_edges(image)

        if use_hough_p:
            # HoughLinesP for line segments
            lines = cv2.HoughLinesP(
                edges,
                self.hough_rho,
                self.hough_theta,
                self.hough_threshold,
                minLineLength=self.hough_min_length,
                maxLineGap=self.hough_max_gap,
            )
        else:
            # HoughLines for infinite lines
            lines = cv2.HoughLines(
                edges,
                self.hough_rho,
                self.hough_theta,
                self.hough_threshold,
            )

        line_segments = []
        if lines is not None:
            for line in lines:
                if use_hough_p:
                    x1, y1, x2, y2 = line[0]
                    line_segments.append(((x1, y1), (x2, y2)))
                else:
                    rho, theta = line[0]
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    line_segments.append(((x1, y1), (x2, y2)))

        logger.info(f"Detected {len(line_segments)} line segments")
        return line_segments

    def filter_lines_by_length(
        self,
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        min_length: float = 30,
        max_length: float = None,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Filter lines by length.

        Args:
            lines: List of line segments
            min_length: Minimum line length
            max_length: Maximum line length (None for no limit)

        Returns:
            Filtered line segments
        """
        filtered = []
        for (x1, y1), (x2, y2) in lines:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= min_length:
                if max_length is None or length <= max_length:
                    filtered.append(((x1, y1), (x2, y2)))
        logger.info(f"Filtered to {len(filtered)} lines (length: {min_length}-{max_length})")
        return filtered

    def detect_intersections(
        self,
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        distance_threshold: float = 5,
    ) -> List[Tuple[float, float]]:
        """
        Detect intersection points between line segments.

        Args:
            lines: List of line segments
            distance_threshold: Maximum distance to consider as intersection

        Returns:
            List of intersection points (x, y)
        """
        intersections = []

        for i, ((x1_1, y1_1), (x2_1, y2_1)) in enumerate(lines):
            for (x1_2, y1_2), (x2_2, y2_2) in lines[i + 1 :]:
                # Calculate intersection point
                x_int, y_int = self._line_intersection(
                    (x1_1, y1_1), (x2_1, y2_1),
                    (x1_2, y1_2), (x2_2, y2_2),
                )

                if x_int is not None:
                    intersections.append((x_int, y_int))

        logger.info(f"Detected {len(intersections)} intersection points")
        return intersections

    @staticmethod
    def _line_intersection(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Calculate intersection of two line segments.

        Args:
            p1, p2: Endpoints of first line
            p3, p4: Endpoints of second line

        Returns:
            Intersection point (x, y) or (None, None) if no intersection
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None, None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return x, y

        return None, None

    def detect_pipes(
        self,
        image: np.ndarray,
        image_path: str = None,
        vlm_labeler: callable = None,
    ) -> Dict[str, Any]:
        """
        Complete pipe detection pipeline.

        Args:
            image: Input image (BGR)
            image_path: Path to image (for OCR pipe label detection)

        Returns:
            Dictionary containing:
                - lines: Detected line segments
                - intersections: Detected intersection points
                - pipe_labels: Detected pipe labels
                - edges: Edge map
        """
        # Preprocess
        preprocessed = ImagePreprocessor.preprocess_for_detection(image)

        # Detect lines
        lines = self.detect_lines(preprocessed)

        # Detect intersections
        intersections = self.detect_intersections(lines)

        # Get edges for visualization
        edges = self.detect_edges(preprocessed)

        # Detect pipe labels via OCR first
        pipe_labels = []
        if image_path:
            pipe_labels = self.detect_pipe_labels(image_path)

        # If no pipe labels found and a VLM labeler is provided, generate candidate
        # crops around intersections and ask VLM to label them.
        if not pipe_labels and vlm_labeler is not None:
            logger.info("No OCR pipe labels found — running VLM labeler on candidates")
            # Create candidate crops around intersections (small bbox around each point)
            cand_size_w = 200
            cand_size_h = 60
            h, w = image.shape[:2]
            for (cx, cy) in intersections:
                x1 = int(max(0, cx - cand_size_w // 2))
                y1 = int(max(0, cy - cand_size_h // 2))
                x2 = int(min(w, cx + cand_size_w // 2))
                y2 = int(min(h, cy + cand_size_h // 2))
                crop = image[y1:y2, x1:x2]
                try:
                    vlm_res = vlm_labeler(crop)
                except Exception as e:
                    logger.error(f"VLM labeler error: {e}")
                    vlm_res = None

                if vlm_res and isinstance(vlm_res, dict):
                    label = vlm_res.get("label")
                    conf = float(vlm_res.get("confidence", 0.0))
                    if label and conf >= 0.3:
                        # Build region dict similar to OCR output
                        region = {
                            "text": label.strip().upper(),
                            "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2},
                            "confidence": conf,
                        }
                        region["type"] = "pipe"
                        pipe_labels.append(region)

        return {
            "lines": lines,
            "intersections": intersections,
            "pipe_labels": pipe_labels,
            "edges": edges,
            "preprocessed_image": preprocessed,
        }

    def visualize_pipes(
        self,
        image: np.ndarray,
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        intersections: List[Tuple[float, float]] = None,
        pipe_labels: List[Dict[str, Any]] = None,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Visualize detected pipes and labels.

        Args:
            image: Original image (BGR)
            lines: Detected line segments
            intersections: Intersection points (optional)
            pipe_labels: Pipe label regions (optional)
            output_path: Path to save visualization (optional)

        Returns:
            Visualization image
        """
        vis_image = image.copy()

        # Draw lines
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw intersections
        if intersections:
            for x, y in intersections:
                cv2.circle(vis_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw pipe labels
        if pipe_labels:
            for label in pipe_labels:
                bbox = label["bbox"]
                x_min = int(bbox["x_min"])
                y_min = int(bbox["y_min"])
                x_max = int(bbox["x_max"])
                y_max = int(bbox["y_max"])
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
                cv2.putText(
                    vis_image,
                    label["text"],
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )

        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Saved pipe visualization to: {output_path}")

        return vis_image
