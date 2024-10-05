import re

def clean_extracted_text(text):
    cleaned_text = re.sub(r"페이지\s*\d+|차례|발행일.*", "", text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text
