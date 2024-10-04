import fitz  # PyMuPDF 라이브러리
import re
import os
import glob

# PDF에서 텍스트를 추출하는 함수
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf_document:
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text

# 추출된 텍스트에서 불필요한 정보를 제거하는 함수
def clean_extracted_text(text):
    # 차례, 페이지 번호 및 발행일 제거
    cleaned_text = re.sub(r"페이지\s*\d+|차례|발행일.*|목차.*", "", text)
    
    # 여러 개의 공백을 하나로 변환
    cleaned_text = ' '.join(cleaned_text.split())
    
    # 불필요한 줄바꿈 제거
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

    return cleaned_text

# 전처리된 데이터를 저장할 폴더 생성
def create_preprocessed_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# 전처리된 텍스트를 파일로 저장하는 함수
def save_preprocessed_text(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

# 절대 경로 사용
pdf_folder = r"C:\Users\dev\Documents\GitHub\sideProject\MTFT\DATA\PDF"  # 절대경로로 설정
preprocessed_folder = r"C:\Users\dev\Documents\GitHub\sideProject\MTFT\DATA\preprocessed_data"
create_preprocessed_folder(preprocessed_folder)

# 경로 확인 출력
print(f"PDF 파일을 찾고 있는 폴더: {pdf_folder}")

# 모든 PDF 파일 처리
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))  # 해당 폴더의 모든 PDF 파일 리스트
print(f"찾은 PDF 파일 목록: {pdf_files}")

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    
    try:
        pdf_text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_extracted_text(pdf_text)
        file_name = os.path.basename(pdf_file).replace(".pdf", "_preprocessed.txt")
        output_file_path = os.path.join(preprocessed_folder, file_name)
        save_preprocessed_text(output_file_path, cleaned_text)
        print(f"전처리된 텍스트가 다음 경로에 저장되었습니다: {output_file_path}")
    except Exception as e:
        print(f"처리 중 에러 발생: {e}")
