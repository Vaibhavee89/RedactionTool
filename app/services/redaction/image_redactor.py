import cv2
import numpy as np
from typing import List, Dict, Any
from app.services.ingestion.image_loader import ImageLoader

class ImageRedactor:
    def __init__(self):
        self.loader = ImageLoader()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def redact_faces(self, image_path: str, output_path: str):
        """
        Redact detected faces in an image.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            roi = image[y:y+h, x:x+w]
            # Dynamic kernel
            k_w, k_h = w, h
            if k_w % 2 == 0: k_w += 1
            if k_h % 2 == 0: k_h += 1
            k_w = min(k_w, 99)
            k_h = min(k_h, 99)
            
            blurred = cv2.GaussianBlur(roi, (k_w, k_h), 30)
            image[y:y+h, x:x+w] = blurred
            
        cv2.imwrite(output_path, image)
        return len(faces)

    def redact_image(self, image_path: str, findings: List[Dict[str, Any]], output_path: str):
        """
        Redact regions in an image corresponding to PII findings.
        
        Args:
            image_path: Path to the input image.
            findings: List of PII findings (from DetectorEngine).
            output_path: Path to save the redacted image.
        """
        # 1. Get OCR data (text + bboxes)
        ocr_data = self.loader.get_ocr_data(image_path)
        
        # 2. Load image for processing
        image = cv2.imread(image_path)
        
        # 3. Match findings to OCR boxes
        # Findings give us character offsets in the full text string.
        # OCR data gives us individual words and their boxes.
        # We need to map the full text offsets back to specific OCR words.
        
        # Reconstruct full text to map offsets
        n_boxes = len(ocr_data['text'])
        full_text = ""
        word_spans = [] # List of (start, end, index_in_ocr)
        
        # Simplification: This mapping assumes pytesseract output order matches simple concatenation
        # In practice, accurate mapping requires careful handling of spaces/newlines
        current_idx = 0
        for i in range(n_boxes):
            word = ocr_data['text'][i]
            conf = int(ocr_data['conf'][i])
            if conf > 0 and word.strip():
                start = current_idx
                end = current_idx + len(word)
                word_spans.append({
                    "start": start,
                    "end": end,
                    "ocr_index": i,
                    "text": word
                })
                full_text += word + " " 
                current_idx = end + 1 # +1 for space
            
        # 4. Blur regions
        for finding in findings:
            # Finding has text, start, end (relative to the full text analyzed by the detector)
            # NOTE: usage of different text extraction methods (simple string vs data) might cause offset mismatch.
            # Best approach for this phase: scan the OCR words and check if they overlap with finding text
            
            f_text = finding['text']
            # Simple word matching for prototype stability
            # Find all words in OCR data that are part of the PII
            
            for span in word_spans:
                # Check if this word is part of the sensitive text
                # Simple check: is the word contained in the finding text?
                # or is the finding text contained in the word?
                
                if span['text'] in f_text or f_text in span['text']:
                    idx = span['ocr_index']
                    x, y, w, h = (ocr_data['left'][idx], ocr_data['top'][idx], 
                                  ocr_data['width'][idx], ocr_data['height'][idx])
                    
                    # ROI to blur
                    roi = image[y:y+h, x:x+w]
                    
                    # Apply Gaussian Blur
                    # dynamic kernel size
                    k_w, k_h = w, h
                    if k_w % 2 == 0: k_w += 1
                    if k_h % 2 == 0: k_h += 1
                    # limit kernel size
                    k_w = min(k_w, 99)
                    k_h = min(k_h, 99)
                    
                    blurred = cv2.GaussianBlur(roi, (k_w, k_h), 30)
                    image[y:y+h, x:x+w] = blurred

        # 5. Save
        cv2.imwrite(output_path, image)
