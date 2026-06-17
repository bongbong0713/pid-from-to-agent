"""Image preprocessing module for P&ID analysis."""

import cv2
import numpy as np
from typing import Tuple, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImagePreprocessor:
    """Preprocessing operations for P&ID images."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to the image file

        Returns:
            Image as numpy array (BGR format)
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
        logger.info(f"Loaded image: {image_path}, shape: {image.shape}")
        return image

    @staticmethod
    def resize_image(
        image: np.ndarray,
        max_width: int = 1920,
        max_height: int = 1080,
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height

        Returns:
            Resized image
        """
        height, width = image.shape[:2]

        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {(width, height)} to {(new_width, new_height)}")

        return image

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: Input image (BGR)

        Returns:
            Grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted image to grayscale")
        return gray

    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Input grayscale image
            clip_limit: Contrast limit

        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        logger.info("Enhanced image contrast")
        return enhanced

    @staticmethod
    def apply_gaussian_blur(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        """
        Apply Gaussian blur.

        Args:
            image: Input image
            kernel_size: Blur kernel size

        Returns:
            Blurred image
        """
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        logger.info(f"Applied Gaussian blur with kernel size: {kernel_size}")
        return blurred

    @staticmethod
    def apply_bilateral_filter(
        image: np.ndarray,
        diameter: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.

        Args:
            image: Input image
            diameter: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space

        Returns:
            Filtered image
        """
        if len(image.shape) == 2:
            # Grayscale
            filtered = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        else:
            # Color
            filtered = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        logger.info("Applied bilateral filter")
        return filtered

    @staticmethod
    def apply_threshold(
        image: np.ndarray,
        method: str = "binary",
        threshold_value: int = 127,
    ) -> np.ndarray:
        """
        Apply thresholding.

        Args:
            image: Input grayscale image
            method: 'binary', 'otsu', or 'adaptive'
            threshold_value: Threshold value (for binary method)

        Returns:
            Thresholded image
        """
        if method == "binary":
            _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "otsu":
            _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        elif method == "adaptive":
            thresholded = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        logger.info(f"Applied {method} thresholding")
        return thresholded

    @staticmethod
    def apply_morphological_operations(
        image: np.ndarray,
        operation: str = "close",
        kernel_size: Tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        """
        Apply morphological operations.

        Args:
            image: Input binary image
            operation: 'open', 'close', 'erode', or 'dilate'
            kernel_size: Kernel size

        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        if operation == "open":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            result = cv2.erode(image, kernel)
        elif operation == "dilate":
            result = cv2.dilate(image, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")

        logger.info(f"Applied {operation} operation")
        return result

    @staticmethod
    def preprocess_for_detection(
        image: np.ndarray,
        resize_max_width: int = 1920,
        resize_max_height: int = 1080,
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for detection.

        Args:
            image: Input image (BGR)
            resize_max_width: Maximum width for resizing
            resize_max_height: Maximum height for resizing

        Returns:
            Preprocessed image
        """
        # Resize
        image = ImagePreprocessor.resize_image(image, resize_max_width, resize_max_height)

        # Convert to grayscale
        gray = ImagePreprocessor.convert_to_grayscale(image)

        # Enhance contrast
        enhanced = ImagePreprocessor.enhance_contrast(gray)

        # Apply bilateral filter
        filtered = ImagePreprocessor.apply_bilateral_filter(enhanced)

        logger.info("Completed preprocessing pipeline")
        return filtered
