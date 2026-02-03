import argparse
import sys
from app.services.pii.detector_engine import DetectorEngine
from app.services.ingestion.pdf_loader import PDFLoader
from app.core.config import Config

def main():
    parser = argparse.ArgumentParser(description="RedactionTool Enterprise CLI")
    parser.add_argument("input_path", help="Path to the file to process")
    parser.add_argument("--mode", choices=["detect", "redact"], default="detect", help="Operation mode")
    
    args = parser.parse_args()
    
    # Init services
    Config.setup_paths()
    engine = DetectorEngine()
    
    # 1. Ingest (Simplistic for now, assuming PDF)
    # in real usage, we'd detect file type
    print(f"Processing: {args.input_path}")
    text = ""
    if args.input_path.lower().endswith(".pdf"):
        loader = PDFLoader()
        text = loader.load(args.input_path)
    else:
        print("Only PDF supported in CLI proof-of-concept for now.")
        return

    # 2. Detect
    print("Running PII detection...")
    results = engine.detect(text)
    
    # 3. Output
    print(f"Found {len(results)} entities.")
    for res in results:
        print(f"- [{res['entity_type']}] {res['text']} ({res['confidence']})")

if __name__ == "__main__":
    main()
