"""
Enhanced Visual Redaction with configurable styles for images and videos.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum


class RedactionStyle(Enum):
    """Visual redaction styles."""
    BLUR = "blur"                    # Gaussian blur
    PIXELATE = "pixelate"            # Pixelation effect
    BLACK_BOX = "black_box"          # Solid black rectangle
    WHITE_BOX = "white_box"          # Solid white rectangle
    COLORED_BOX = "colored_box"      # Custom colored rectangle
    PATTERN = "pattern"              # Pattern fill (diagonal lines, etc.)
    HEAVY_BLUR = "heavy_blur"        # Extra strong blur
    MOSAIC = "mosaic"                # Mosaic effect


class VisualRedactor:
    """
    Enhanced visual redactor with multiple configurable styles.

    Supports:
    - Multiple redaction styles (blur, pixelate, boxes, patterns)
    - Configurable intensity/strength
    - Per-region custom styles
    - Face detection and redaction
    - Bounding box redaction for PII text regions
    - Video frame processing
    """

    def __init__(
        self,
        default_style: RedactionStyle = RedactionStyle.BLUR,
        default_color: Tuple[int, int, int] = (0, 0, 0),
        blur_strength: int = 30,
        pixelate_size: int = 10
    ):
        """
        Initialize visual redactor.

        Args:
            default_style: Default redaction style
            default_color: Default color for colored boxes (BGR format)
            blur_strength: Blur kernel strength (higher = more blur)
            pixelate_size: Pixelation block size
        """
        self.default_style = default_style
        self.default_color = default_color
        self.blur_strength = blur_strength
        self.pixelate_size = pixelate_size

        # Load face cascade for face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.face_cascade = None

    def redact_region(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None,
        intensity: Optional[int] = None
    ) -> np.ndarray:
        """
        Redact a specific region in the image.

        Args:
            image: Input image (numpy array)
            x, y, w, h: Region coordinates (x, y, width, height)
            style: Redaction style (None = use default)
            color: Color for colored box style
            intensity: Redaction intensity (style-specific)

        Returns:
            Image with region redacted
        """
        style = style or self.default_style
        color = color or self.default_color

        # Extract ROI
        roi = image[y:y+h, x:x+w]

        if style == RedactionStyle.BLUR:
            redacted_roi = self._apply_blur(roi, intensity or self.blur_strength)

        elif style == RedactionStyle.HEAVY_BLUR:
            redacted_roi = self._apply_blur(roi, intensity or (self.blur_strength * 2))

        elif style == RedactionStyle.PIXELATE:
            redacted_roi = self._apply_pixelate(roi, intensity or self.pixelate_size)

        elif style == RedactionStyle.MOSAIC:
            redacted_roi = self._apply_mosaic(roi, intensity or self.pixelate_size)

        elif style == RedactionStyle.BLACK_BOX:
            redacted_roi = np.zeros_like(roi)

        elif style == RedactionStyle.WHITE_BOX:
            redacted_roi = np.full_like(roi, 255)

        elif style == RedactionStyle.COLORED_BOX:
            redacted_roi = np.full_like(roi, color)

        elif style == RedactionStyle.PATTERN:
            redacted_roi = self._apply_pattern(roi)

        else:
            # Default to blur
            redacted_roi = self._apply_blur(roi, self.blur_strength)

        # Replace ROI in original image
        image[y:y+h, x:x+w] = redacted_roi

        return image

    def _apply_blur(self, roi: np.ndarray, strength: int) -> np.ndarray:
        """Apply Gaussian blur to region."""
        h, w = roi.shape[:2]

        # Calculate kernel size (must be odd)
        k_w = min(w, strength * 2 + 1)
        k_h = min(h, strength * 2 + 1)
        if k_w % 2 == 0:
            k_w += 1
        if k_h % 2 == 0:
            k_h += 1

        # Limit kernel size
        k_w = min(k_w, 99)
        k_h = min(k_h, 99)

        return cv2.GaussianBlur(roi, (k_w, k_h), strength)

    def _apply_pixelate(self, roi: np.ndarray, block_size: int) -> np.ndarray:
        """Apply pixelation effect."""
        h, w = roi.shape[:2]

        if h == 0 or w == 0:
            return roi

        # Resize down then up to create pixelation
        small_h = max(1, h // block_size)
        small_w = max(1, w // block_size)

        # Downsample
        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Upsample back to original size
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        return pixelated

    def _apply_mosaic(self, roi: np.ndarray, block_size: int) -> np.ndarray:
        """Apply mosaic effect (similar to pixelate but with averaging)."""
        h, w = roi.shape[:2]

        if h == 0 or w == 0:
            return roi

        # Create mosaic by averaging blocks
        mosaic = roi.copy()

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Get block boundaries
                i_end = min(i + block_size, h)
                j_end = min(j + block_size, w)

                # Calculate average color of block
                block = roi[i:i_end, j:j_end]
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)

                # Fill block with average color
                mosaic[i:i_end, j:j_end] = avg_color

        return mosaic

    def _apply_pattern(self, roi: np.ndarray) -> np.ndarray:
        """Apply diagonal line pattern."""
        h, w = roi.shape[:2]
        pattern = np.zeros_like(roi)

        # Draw diagonal lines
        for i in range(-h, w, 10):
            cv2.line(pattern, (i, 0), (i + h, h), (255, 255, 255), 2)

        return pattern

    def redact_faces(
        self,
        image: np.ndarray,
        style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Detect and redact faces in image.

        Args:
            image: Input image
            style: Redaction style
            color: Color for colored boxes

        Returns:
            Tuple of (redacted image, number of faces found)
        """
        if self.face_cascade is None:
            raise RuntimeError("Face cascade not loaded")

        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Redact each face
        for (x, y, w, h) in faces:
            image = self.redact_region(image, x, y, w, h, style, color)

        return image, len(faces)

    def redact_bounding_boxes(
        self,
        image: np.ndarray,
        bounding_boxes: List[Dict[str, Any]],
        style_map: Optional[Dict[str, RedactionStyle]] = None,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Redact multiple bounding boxes with optional per-entity styling.

        Args:
            image: Input image
            bounding_boxes: List of bounding box dicts with keys:
                - 'x', 'y', 'width', 'height' or 'left', 'top', 'width', 'height'
                - 'entity_type' (optional) for style mapping
            style_map: Map entity_type to RedactionStyle
            color_map: Map entity_type to color

        Returns:
            Redacted image
        """
        for bbox in bounding_boxes:
            # Extract coordinates (support both formats)
            x = bbox.get('x', bbox.get('left', 0))
            y = bbox.get('y', bbox.get('top', 0))
            w = bbox.get('width', bbox.get('w', 0))
            h = bbox.get('height', bbox.get('h', 0))

            # Get entity-specific style
            entity_type = bbox.get('entity_type')
            style = None
            color = None

            if entity_type and style_map:
                style = style_map.get(entity_type)

            if entity_type and color_map:
                color = color_map.get(entity_type)

            # Redact region
            image = self.redact_region(image, x, y, w, h, style, color)

        return image

    def redact_image_file(
        self,
        input_path: str,
        output_path: str,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        redact_faces: bool = False,
        style: Optional[RedactionStyle] = None,
        style_map: Optional[Dict[str, RedactionStyle]] = None
    ) -> Dict[str, Any]:
        """
        Redact image file with bounding boxes and/or faces.

        Args:
            input_path: Input image path
            output_path: Output image path
            bounding_boxes: List of bounding boxes to redact
            redact_faces: Whether to detect and redact faces
            style: Default style for all redactions
            style_map: Per-entity style mapping

        Returns:
            Dictionary with redaction results
        """
        # Load image
        image = cv2.imread(input_path)

        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        faces_count = 0
        bbox_count = 0

        # Redact faces
        if redact_faces:
            image, faces_count = self.redact_faces(image, style)

        # Redact bounding boxes
        if bounding_boxes:
            image = self.redact_bounding_boxes(image, bounding_boxes, style_map)
            bbox_count = len(bounding_boxes)

        # Save redacted image
        cv2.imwrite(output_path, image)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'faces_redacted': faces_count,
            'regions_redacted': bbox_count,
            'total_redactions': faces_count + bbox_count,
            'success': True
        }

    def redact_video_file(
        self,
        input_path: str,
        output_path: str,
        redact_faces: bool = True,
        style: Optional[RedactionStyle] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Redact video file frame by frame.

        Args:
            input_path: Input video path
            output_path: Output video path
            redact_faces: Whether to detect and redact faces
            style: Redaction style
            progress_callback: Optional callback(float) for progress (0.0 to 1.0)

        Returns:
            Dictionary with redaction results
        """
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine codec based on file extension
        ext = output_path.split('.')[-1].lower()
        if ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == 'mov':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_faces = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Redact faces in frame
            if redact_faces:
                frame, faces = self.redact_faces(frame, style)
                total_faces += faces

            out.write(frame)

            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)

        cap.release()
        out.release()

        return {
            'input_path': input_path,
            'output_path': output_path,
            'frames_processed': frame_count,
            'faces_redacted': total_faces,
            'fps': fps,
            'success': True
        }


class StyleConfig:
    """
    Configuration for redaction styles with policy integration.
    """

    def __init__(self):
        """Initialize style configuration."""
        self.entity_styles: Dict[str, Dict[str, Any]] = {}
        self.default_text_style: str = 'block'
        self.default_visual_style: RedactionStyle = RedactionStyle.BLUR

    def set_entity_style(
        self,
        entity_type: str,
        text_style: Optional[str] = None,
        visual_style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None,
        intensity: Optional[int] = None
    ):
        """
        Set redaction style for entity type.

        Args:
            entity_type: Entity type (e.g., 'PAN', 'PERSON')
            text_style: Text redaction style ('block', 'mask', 'label', etc.)
            visual_style: Visual redaction style
            color: Color for visual redaction
            intensity: Intensity for visual effects
        """
        self.entity_styles[entity_type] = {
            'text_style': text_style or self.default_text_style,
            'visual_style': visual_style or self.default_visual_style,
            'color': color,
            'intensity': intensity
        }

    def get_entity_style(self, entity_type: str) -> Dict[str, Any]:
        """Get style configuration for entity type."""
        return self.entity_styles.get(entity_type, {
            'text_style': self.default_text_style,
            'visual_style': self.default_visual_style,
            'color': None,
            'intensity': None
        })

    def load_from_dict(self, config: Dict[str, Any]):
        """Load style configuration from dictionary."""
        self.default_text_style = config.get('default_text_style', 'block')
        self.default_visual_style = RedactionStyle(
            config.get('default_visual_style', 'blur')
        )

        for entity_type, style_config in config.get('entity_styles', {}).items():
            visual_style = style_config.get('visual_style')
            if isinstance(visual_style, str):
                visual_style = RedactionStyle(visual_style)

            self.set_entity_style(
                entity_type,
                text_style=style_config.get('text_style'),
                visual_style=visual_style,
                color=style_config.get('color'),
                intensity=style_config.get('intensity')
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert style configuration to dictionary."""
        return {
            'default_text_style': self.default_text_style,
            'default_visual_style': self.default_visual_style.value,
            'entity_styles': {
                entity_type: {
                    'text_style': config['text_style'],
                    'visual_style': config['visual_style'].value,
                    'color': config['color'],
                    'intensity': config['intensity']
                }
                for entity_type, config in self.entity_styles.items()
            }
        }
