from langdetect import detect

def detect_language(text: str) -> str:
    """Detect the language of the text"""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English
