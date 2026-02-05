"""
Image preprocessing module for OCR enhancement.
Includes de-skewing, noise removal, and contrast enhancement.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import math


class ImagePreprocessor:
    """
    Advanced image preprocessing for improved OCR accuracy.

    Features:
    - De-skewing (automatic rotation correction)
    - Noise removal (denoising)
    - Contrast enhancement (adaptive histogram equalization)
    - Binarization (adaptive thresholding)
    - Border removal
    """

    def __init__(self):
        pass

    def preprocess(
        self,
        image: np.ndarray,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        binarize: bool = True,
        remove_borders: bool = False
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.

        Args:
            image: Input image (numpy array)
            deskew: Apply deskewing
            denoise: Apply noise removal
            enhance_contrast: Apply contrast enhancement
            binarize: Apply binarization
            remove_borders: Remove image borders

        Returns:
            Preprocessed image
        """
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Remove borders
        if remove_borders:
            processed = self.remove_border(processed)

        # Denoise
        if denoise:
            processed = self.denoise_image(processed)

        # Enhance contrast
        if enhance_contrast:
            processed = self.enhance_contrast(processed)

        # Deskew
        if deskew:
            processed = self.deskew_image(processed)

        # Binarize
        if binarize:
            processed = self.binarize_image(processed)

        return processed

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically detect and correct image skew.
        Uses Hough Line Transform to detect document orientation.

        Args:
            image: Grayscale input image

        Returns:
            Deskewed image
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return image

        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)

        if not angles:
            return image

        # Get median angle
        median_angle = np.median(angles)

        # Only rotate if angle is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            return image

        # Rotate image
        return self.rotate_image(image, median_angle)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size to avoid cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Rotate
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            borderValue=(255, 255, 255)
        )

        return rotated

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image using Non-local Means Denoising.

        Args:
            image: Grayscale input image

        Returns:
            Denoised image
        """
        # Use fastNlMeansDenoising for grayscale images
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=10,  # Filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Grayscale input image

        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary (black and white) using adaptive thresholding.

        Args:
            image: Grayscale input image

        Returns:
            Binary image
        """
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        return binary

    def remove_border(self, image: np.ndarray, border_size: int = 10) -> np.ndarray:
        """
        Remove borders from image.

        Args:
            image: Input image
            border_size: Size of border to remove (pixels)

        Returns:
            Image with borders removed
        """
        height, width = image.shape[:2]

        if height <= 2 * border_size or width <= 2 * border_size:
            return image

        return image[border_size:height-border_size, border_size:width-border_size]

    def resize_for_ocr(
        self,
        image: np.ndarray,
        target_height: int = 1000,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image to optimal size for OCR.

        Args:
            image: Input image
            target_height: Target height in pixels
            maintain_aspect: Maintain aspect ratio

        Returns:
            Resized image
        """
        height, width = image.shape[:2]

        if height >= target_height:
            return image

        if maintain_aspect:
            scale = target_height / height
            new_width = int(width * scale)
            new_height = target_height
        else:
            new_width = width
            new_height = target_height

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return resized

    def preprocess_from_path(
        self,
        image_path: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image from file path.

        Args:
            image_path: Path to image file
            **kwargs: Preprocessing options

        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess
        preprocessed = self.preprocess(image, **kwargs)

        return image, preprocessed

    def preprocess_from_pil(
        self,
        pil_image: Image.Image,
        **kwargs
    ) -> np.ndarray:
        """
        Preprocess PIL Image.

        Args:
            pil_image: PIL Image object
            **kwargs: Preprocessing options

        Returns:
            Preprocessed image as numpy array
        """
        # Convert PIL to numpy array
        image = np.array(pil_image)

        # Convert RGB to BGR (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocess
        preprocessed = self.preprocess(image, **kwargs)

        return preprocessed

    def save_preprocessed(self, image: np.ndarray, output_path: str):
        """
        Save preprocessed image to file.

        Args:
            image: Preprocessed image
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
