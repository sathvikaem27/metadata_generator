import re
import os
import fitz  # PyMuPDF
import docx
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import nltk
from transformers import pipeline
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize

# --- NLTK Resource Download Check ---
# It's good practice to encapsulate this in a function or a setup script,
# but for a single file, this is fine.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' resource not found. Attempting to download...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' resource downloaded successfully.")

# --- Load Models (Singleton pattern is a good practice for larger apps) ---
print("Loading NLP models. This may take a moment...")
try:
    # Using a more robust model for summarization can sometimes yield better results.
    # facebook/bart-large-cnn is a great choice.
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure you have a working internet connection and the required libraries installed.")
    # Exit if models can't be loaded, as the script is unusable.
    exit()

# --- Generalized Non-Content Patterns ---
# Focus on structure and common document artifacts rather than specific content.
COMMON_NON_CONTENT_PATTERNS = [
    r'^\s*page\s*\d+\s*of\s*\d+\s*$',       # "Page 1 of 10"
    r'^\s*table\s+of\s+contents\s*$',      # "Table of Contents"
    r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$',     # Dates like "12/25/2024"
    r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s\d{1,2},\s\d{4}$', # "Jan 1, 2024"
    r'^\s*confidential\s*$',               # "Confidential"
    r'^\s*internal\s+use\s+only\s*$',      # "Internal Use Only"
    r'appendix-[a-z]',                     # "appendix-a"
    r'^https?://[^\s]+$',                   # Lines containing only a URL
]

# ----------- EXTRACT TEXT FROM FILE (Your original function is solid) -----------
def extract_text(file_path):
    """Extracts raw text from various file formats, including OCR for image-based PDFs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                text_list = []
                for page in doc:
                    page_text = page.get_text()
                    if page_text.strip():  # Extracted text is available
                        text_list.append(page_text)
                    else:
                        # 🧠 OCR fallback if no extractable text
                        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        text_list.append(ocr_text)
                text = "\n".join(text_list)
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
        elif ext in [".jpg", ".jpeg", ".png"]:
            text = pytesseract.image_to_string(Image.open(file_path))
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""  # Return empty string on failure

    return text


# ----------- GET TITLE (Improved Logic) -----------
def get_title(text, fallback_name="Untitled Document"):
    """
    Extracts a document title from the first few lines of text.
    It prioritizes shorter, title-cased lines.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Search the first 10 lines for a plausible title
    for line in lines[:10]:
        # A good title is usually short (2-15 words)
        if 2 < len(line.split()) < 15 and line.isupper():
            return line
        # Title Case is also a strong indicator
        if 2 < len(line.split()) < 15 and line.istitle():
            return line

    # Fallback: Use the first non-trivial line if it's reasonable
    for line in lines[:10]:
        if 2 < len(line.split()) < 20:
             # Avoid common headers that are not titles
            if not any(keyword in line.lower() for keyword in ['introduction', 'abstract', 'objective', 'summary']):
                return line

    return fallback_name

# ----------- PREPROCESS TEXT FOR ML (Improved Logic) -----------
def preprocess_text_for_ml(text):
    """
    Cleans and prepares text for summarization and keyword extraction models.
    Focuses on removing noise while preserving coherent paragraph structures.
    """
    # 1. Basic cleaning of URLs, emails, and repeated dots
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # 2. Normalize whitespace and fix broken sentences
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Re-join hyphenated words
    text = re.sub(r'\n+', '\n', text).strip()   # Remove excess blank lines

    lines = text.split('\n')
    cleaned_lines = []
    
    # 3. Filter out structural noise and artifacts
    for line in lines:
        line_lower = line.strip().lower()
        if not line_lower:
            continue
        # Check against regex patterns for common non-content lines
        if any(re.search(pattern, line_lower) for pattern in COMMON_NON_CONTENT_PATTERNS):
            continue
        cleaned_lines.append(line.strip())

    # 4. Re-join the cleaned lines into a single block of text
    full_text = " ".join(cleaned_lines)
    # Final whitespace normalization
    full_text = re.sub(r'\s+', ' ', full_text)
    
    return full_text

# ----------- MAIN METADATA GENERATOR (Improved) -----------
def generate_metadata(file_path):
    """
    Generates a title, summary, and keywords for a given file.

    Args:
        file_path (str): The path to the document.

    Returns:
        dict: A dictionary containing 'title', 'keywords', and 'summary'.
    """
    print(f"\nProcessing file: {file_path}")
    # 1. Extract raw text
    raw_text = extract_text(file_path)
    if not raw_text or len(raw_text.strip()) < 50:
        print("Document is empty or too short for processing.")
        return {
            "title": get_title("", fallback_name=os.path.basename(file_path)),
            "keywords": [],
            "summary": "Not enough content to generate a summary."
        }

    # 2. Get title from the *raw* text for best results
    title = get_title(raw_text, fallback_name=os.path.basename(file_path))

    # 3. Preprocess the text specifically for ML models
    ml_ready_text = preprocess_text_for_ml(raw_text)

    # --- Generate Summary ---
    summary = "Could not generate summary."
    try:
        # Ensure the text is substantial enough for the model
        if len(ml_ready_text.split()) > 60:
            summary_out = summarizer(ml_ready_text, max_length=150, min_length=40, do_sample=False)
            summary = summary_out[0]['summary_text']
        else:
            # Fallback for short texts: use the first few sentences
            summary = ". ".join(sent_tokenize(ml_ready_text)[:3])
    except Exception as e:
        print(f"[Summarization Error] Could not use ML model: {e}")
        # A simple fallback if the ML model fails for any reason
        summary = ". ".join(sent_tokenize(ml_ready_text)[:3]) + "..." if ml_ready_text else summary

    # --- Generate Keywords ---
    keywords = []
    try:
        # Use a slightly longer text for keywords to get better context
        text_for_keywords = ml_ready_text[:5000]
        if len(text_for_keywords.split()) > 10:
            extracted_keywords = kw_model.extract_keywords(
                text_for_keywords,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=20,
                top_n=5
            )
            keywords = [kw[0] for kw in extracted_keywords]
    except Exception as e:
        print(f"[Keyword Error] {e}. No keywords generated.")

    return {
        "title": title.strip(),
        "keywords": keywords,
        "summary": summary.strip()
    }
