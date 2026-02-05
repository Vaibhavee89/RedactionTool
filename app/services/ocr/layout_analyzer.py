"""
Layout-aware text extraction for documents.
Detects and preserves document structure including paragraphs, tables, and sections.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BlockType(Enum):
    """Types of document blocks."""
    TEXT = "text"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class DocumentBlock:
    """Represents a block of content in a document."""
    type: BlockType
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    block_num: int
    children: List['DocumentBlock'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class LayoutAnalyzer:
    """
    Analyze document layout and extract structured content.

    Features:
    - Paragraph detection
    - Table detection and extraction
    - Heading detection
    - List detection
    - Reading order detection
    """

    def __init__(self, ocr_engine=None):
        """
        Initialize layout analyzer.

        Args:
            ocr_engine: OCR engine to use (optional)
        """
        self.ocr_engine = ocr_engine

    def analyze_layout(
        self,
        image_input,
        detect_tables: bool = True,
        detect_paragraphs: bool = True,
        detect_headings: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze document layout and extract structured content.

        Args:
            image_input: Image path, PIL Image, or numpy array
            detect_tables: Enable table detection
            detect_paragraphs: Enable paragraph detection
            detect_headings: Enable heading detection

        Returns:
            Dictionary with structured layout information
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
            image = np.array(pil_image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # Get OCR data
        ocr_data = pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT
        )

        # Extract blocks
        blocks = self._extract_blocks(ocr_data)

        # Detect structure
        structured_blocks = []

        if detect_paragraphs:
            paragraphs = self._detect_paragraphs(blocks, ocr_data)
            structured_blocks.extend(paragraphs)

        if detect_tables:
            tables = self._detect_tables(image, ocr_data)
            structured_blocks.extend(tables)

        if detect_headings:
            headings = self._detect_headings(blocks, ocr_data)
            structured_blocks.extend(headings)

        # Sort by reading order (top to bottom, left to right)
        structured_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        return {
            'blocks': structured_blocks,
            'text_by_type': self._group_by_type(structured_blocks),
            'full_text': self._reconstruct_text(structured_blocks),
            'layout_metadata': {
                'num_blocks': len(structured_blocks),
                'num_paragraphs': sum(1 for b in structured_blocks if b.type == BlockType.PARAGRAPH),
                'num_tables': sum(1 for b in structured_blocks if b.type == BlockType.TABLE),
                'num_headings': sum(1 for b in structured_blocks if b.type == BlockType.HEADING)
            }
        }

    def _extract_blocks(self, ocr_data: Dict) -> List[Dict]:
        """Extract blocks from OCR data."""
        blocks = {}

        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                block_num = ocr_data['block_num'][i]
                par_num = ocr_data['par_num'][i]
                line_num = ocr_data['line_num'][i]

                key = (block_num, par_num, line_num)

                if key not in blocks:
                    blocks[key] = {
                        'text': [],
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'conf': [],
                        'block_num': block_num,
                        'par_num': par_num,
                        'line_num': line_num
                    }

                blocks[key]['text'].append(ocr_data['text'][i])
                blocks[key]['conf'].append(ocr_data['conf'][i])

                # Update bbox to include all words
                right = ocr_data['left'][i] + ocr_data['width'][i]
                bottom = ocr_data['top'][i] + ocr_data['height'][i]

                blocks[key]['left'] = min(blocks[key]['left'], ocr_data['left'][i])
                blocks[key]['top'] = min(blocks[key]['top'], ocr_data['top'][i])
                blocks[key]['width'] = max(right - blocks[key]['left'], blocks[key]['width'])
                blocks[key]['height'] = max(bottom - blocks[key]['top'], blocks[key]['height'])

        # Convert to list
        block_list = []
        for key, block in blocks.items():
            block['text'] = ' '.join(block['text'])
            block['conf'] = np.mean(block['conf']) if block['conf'] else 0
            block_list.append(block)

        return block_list

    def _detect_paragraphs(self, blocks: List[Dict], ocr_data: Dict) -> List[DocumentBlock]:
        """Detect paragraphs from blocks."""
        paragraphs = []

        # Group blocks by paragraph number
        par_groups = {}
        for block in blocks:
            par_key = (block['block_num'], block['par_num'])
            if par_key not in par_groups:
                par_groups[par_key] = []
            par_groups[par_key].append(block)

        # Create paragraph blocks
        for par_key, par_blocks in par_groups.items():
            # Combine text from all lines in paragraph
            text_lines = [b['text'] for b in par_blocks]
            text = '\n'.join(text_lines)

            # Calculate combined bounding box
            left = min(b['left'] for b in par_blocks)
            top = min(b['top'] for b in par_blocks)
            right = max(b['left'] + b['width'] for b in par_blocks)
            bottom = max(b['top'] + b['height'] for b in par_blocks)

            conf = np.mean([b['conf'] for b in par_blocks])

            # Determine if it's a heading (larger font, shorter text, etc.)
            is_heading = self._is_heading(par_blocks)

            block_type = BlockType.HEADING if is_heading else BlockType.PARAGRAPH

            paragraphs.append(DocumentBlock(
                type=block_type,
                text=text,
                bbox=(left, top, right - left, bottom - top),
                confidence=conf,
                block_num=par_key[0]
            ))

        return paragraphs

    def _is_heading(self, blocks: List[Dict]) -> bool:
        """
        Determine if blocks represent a heading.

        Heuristics:
        - Single line or very short
        - Larger font (approximated by height)
        - High confidence
        """
        if len(blocks) > 3:
            return False

        # Check if text is short
        total_text = ' '.join([b['text'] for b in blocks])
        if len(total_text) > 100:
            return False

        # Check if at top of page (crude heuristic)
        avg_top = np.mean([b['top'] for b in blocks])
        if avg_top < 100:
            return True

        # Check height (headings usually taller)
        avg_height = np.mean([b['height'] for b in blocks])
        if avg_height > 30:
            return True

        return False

    def _detect_tables(self, image: np.ndarray, ocr_data: Dict) -> List[DocumentBlock]:
        """
        Detect tables in document using line detection.

        Args:
            image: Document image
            ocr_data: OCR data

        Returns:
            List of table blocks
        """
        tables = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply threshold
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (tables are usually large)
            if w > 100 and h > 100:
                # Extract text from table region
                table_text = self._extract_table_text(ocr_data, (x, y, w, h))

                if table_text:
                    tables.append(DocumentBlock(
                        type=BlockType.TABLE,
                        text=table_text,
                        bbox=(x, y, w, h),
                        confidence=0.8,  # Table detection confidence
                        block_num=-1  # Special marker for tables
                    ))

        return tables

    def _extract_table_text(self, ocr_data: Dict, bbox: Tuple[int, int, int, int]) -> str:
        """Extract text from table region."""
        x, y, w, h = bbox
        table_words = []

        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                word_x = ocr_data['left'][i]
                word_y = ocr_data['top'][i]
                word_w = ocr_data['width'][i]
                word_h = ocr_data['height'][i]

                # Check if word is inside table bbox
                if (x <= word_x <= x + w and
                    y <= word_y <= y + h and
                    word_x + word_w <= x + w and
                    word_y + word_h <= y + h):

                    table_words.append({
                        'text': ocr_data['text'][i],
                        'x': word_x,
                        'y': word_y,
                        'line': ocr_data['line_num'][i]
                    })

        if not table_words:
            return ""

        # Sort by line and x position
        table_words.sort(key=lambda w: (w['line'], w['x']))

        # Group by line
        lines = {}
        for word in table_words:
            line_num = word['line']
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(word['text'])

        # Join into table text
        table_lines = []
        for line_num in sorted(lines.keys()):
            table_lines.append(' | '.join(lines[line_num]))

        return '\n'.join(table_lines)

    def _detect_headings(self, blocks: List[Dict], ocr_data: Dict) -> List[DocumentBlock]:
        """Detect headings (already handled in _detect_paragraphs)."""
        # Headings are detected in _detect_paragraphs
        return []

    def _group_by_type(self, blocks: List[DocumentBlock]) -> Dict[str, List[str]]:
        """Group blocks by type."""
        grouped = {}

        for block in blocks:
            type_name = block.type.value
            if type_name not in grouped:
                grouped[type_name] = []
            grouped[type_name].append(block.text)

        return grouped

    def _reconstruct_text(self, blocks: List[DocumentBlock]) -> str:
        """Reconstruct full text from blocks in reading order."""
        text_parts = []

        for block in blocks:
            if block.type == BlockType.HEADING:
                text_parts.append(f"\n## {block.text}\n")
            elif block.type == BlockType.TABLE:
                text_parts.append(f"\n[TABLE]\n{block.text}\n[/TABLE]\n")
            elif block.type == BlockType.PARAGRAPH:
                text_parts.append(f"{block.text}\n")
            else:
                text_parts.append(block.text)

        return '\n'.join(text_parts)

    def extract_tables(self, image_input) -> List[Dict[str, Any]]:
        """
        Extract only tables from document.

        Args:
            image_input: Image to analyze

        Returns:
            List of table dictionaries with text and structure
        """
        result = self.analyze_layout(image_input, detect_tables=True, detect_paragraphs=False)

        tables = [
            {
                'text': block.text,
                'bbox': block.bbox,
                'confidence': block.confidence
            }
            for block in result['blocks']
            if block.type == BlockType.TABLE
        ]

        return tables

    def extract_paragraphs(self, image_input) -> List[str]:
        """
        Extract only paragraphs from document.

        Args:
            image_input: Image to analyze

        Returns:
            List of paragraph texts
        """
        result = self.analyze_layout(image_input, detect_paragraphs=True, detect_tables=False)

        paragraphs = [
            block.text
            for block in result['blocks']
            if block.type == BlockType.PARAGRAPH
        ]

        return paragraphs
