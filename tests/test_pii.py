from app.services.pii.detector_engine import DetectorEngine
import sys

def test_engine():
    print("Initializing Engine...")
    engine = DetectorEngine()
    
    text = "My PAN is ABCDE1234F and email is test@example.com. I live in Mumbai."
    print(f"Analyzing text: {text}")
    
    results = engine.detect(text)
    
    print(f"Found {len(results)} entities:")
    for res in results:
        print(f"- {res['entity_type']}: {res['text']} (Source: {res['source']})")

if __name__ == "__main__":
    test_engine()
