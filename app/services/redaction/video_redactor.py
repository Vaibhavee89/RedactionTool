import cv2
import tempfile
import os

class VideoRedactor:
    def __init__(self):
        # Load face cascade classifier
        # Use cv2 internal path or local file
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def redact_faces(self, video_path: str, output_path: str, progress_callback=None):
        """
        Process video frames to blur faces.
        
        Args:
            video_path: Path to input video.
            output_path: Path to save redacted video.
            progress_callback: Optional callable(float) for progress steps (0.0 to 1.0).
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        # 'mp4v' is widely compatible
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Blur faces
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                # Dynamic kernel
                k_size = (99, 99)
                blurred_face = cv2.GaussianBlur(face_region, k_size, 30)
                frame[y:y+h, x:x+w] = blurred_face
            
            out.write(frame)
            
            # Update progress
            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        out.release()
