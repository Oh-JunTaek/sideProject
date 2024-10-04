import fitz  # PyMuPDF 라이브러리

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf_document:
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text

pdf_text = extract_text_from_pdf("your_chemistry_curriculum.pdf")
print(pdf_text)
