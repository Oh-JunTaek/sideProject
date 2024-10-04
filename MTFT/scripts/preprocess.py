from utils.pdf_extractor import extract_text_from_pdf

pdf_path = "data/your_pdf_files/chemistry_curriculum.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)
