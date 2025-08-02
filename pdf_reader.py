import fitz  # PyMuPDF
import requests
import re

def extract_sentences_from_pdf_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF from {pdf_url}")
    
    pdf_bytes = response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    return re.split(r'(?<=[.?!])\s+', full_text.strip())
