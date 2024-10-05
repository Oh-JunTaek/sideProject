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

# 특수문자 제거 함수
def remove_special_characters(text):
    """
    텍스트에서 특수문자 및 불필요한 기호 제거
    """
    # 유니코드 범위를 사용해 한글, 영어, 숫자 및 공백 외 문자 제거
    cleaned_text = re.sub(r'[^\uAC00-\uD7A3\u0020-\u007E\u3131-\u3163\u1100-\u11FF]', '', text)
    
    # 여러 개의 공백을 하나로 변환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# 추출된 텍스트에서 불필요한 메타데이터를 제거하는 함수
def clean_extracted_text(text):
    # 1. 페이지 번호 제거 (예: 페이지 1, Page 1, 쪽 번호 등)
    cleaned_text = re.sub(r"(페이지|Page|쪽)\s*\d+", "", text)
    
    # 2. 차례 및 목차 제거 (차례, 목차, Contents 등)
    cleaned_text = re.sub(r"(차례|목차|Contents).*", "", cleaned_text)
    
    # 3. 발행일, ISBN, 고시번호, 연락처(전화, 팩스) 등 제거
    cleaned_text = re.sub(r"(발행일|ISBN|고시\s*제.*호|전화\s*[^\n]*|팩스\s*[^\n]*)", "", cleaned_text)
    
    # 4. 연구진, 연구책임자, 협력기관 관련 정보 제거
    cleaned_text = re.sub(r"(연구진\s*[^\n]*|연구책임자\s*[^\n]*|연구개발진\s*[^\n]*|협력기관\s*[^\n]*|연구보고서\s*[^\n]*)", "", cleaned_text)
    
    # 5. 이메일 및 웹사이트 주소 제거
    cleaned_text = re.sub(r"\S+@\S+\.\S+|http\S+", "", cleaned_text)
    
    # 6. 불필요한 구문 제거 (예: 발행, 서지정보, 출판사 등)
    cleaned_text = re.sub(r"(발행\s*[^\n]*|서지\s*[^\n]*|출판\s*[^\n]*|출판사\s*[^\n]*)", "", cleaned_text)

    # 7. 연구보고서 성과 및 관련 구문 제거
    cleaned_text = re.sub(r"(정책\s*제안|성과\s*발표|연구\s*결과|연구\s*목표|프로젝트\s*성과).*", "", cleaned_text)

    # 8. 표 및 그림 번호 제거 (예: 표 1, 그림 2 등)
    cleaned_text = re.sub(r"(표|그림)\s*\d+", "", cleaned_text)

    # 9. 불필요한 줄바꿈 제거 (빈 줄은 삭제하고, 여러 줄바꿈은 한 줄로 통합)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    
    # 10. 여러 개의 공백을 하나로 변환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # 11. 특수문자 제거 (기존의 특수문자 제거 패턴 유지)
    cleaned_text = remove_special_characters(cleaned_text)
    
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
