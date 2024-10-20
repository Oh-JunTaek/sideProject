from utils.md_file_manager import handle_content
from utils.schedule_manager import handle_schedule_content
import os
from utils.input_processor import process_input

# 사용자 입력을 처리하고 전체 흐름을 진행하는 함수
def main():
    # 1. 사용자로부터 공지사항 입력 받기
    print("공지사항을 입력하세요:")
    user_input_text = input()

    # 2. JSON 파일 경로 설정
    json_file = "utils/keywords.json"

    # 3. 공지사항에서 title과 content 추출
    title, content = process_input(user_input_text)

    # 추출된 title과 content 확인
    print(f"추출된 Title: {title}")
    print(f"추출된 Content: {content}")

    # 4. JSON 기반으로 키워드와 카테고리 분류 (LLM 제거)
    print("공지 내용을 처리 중입니다...")

    # 5. 스케줄 확인 절차
    schedule_confirm = None
    if "schedule" in content.lower():  # 스케줄 관련 정보가 있을 경우
        print(f"Detected Schedule: {title}")
        schedule_confirm = "y"
    else:
        schedule_confirm = "n"

    # 6. 사용자가 분류한 카테고리를 입력받기 (자동 추출 없이 수동으로 선택)
    print("해당 공지의 카테고리를 숫자로 입력하세요. 1. money 2. management_edu 3. others")
    category_input = input()
    category_mapping = {
        "1": "money",
        "2": "management_edu",
        "3": "others"
    }
    new_category = category_mapping.get(category_input, "others")

    # 7. 스케줄 확인 후 내용 처리
    if schedule_confirm == "y":
        handle_schedule_content(user_input_text)

    # 8. 최종적으로 선택한 카테고리로 md 파일에 내용 추가
    handle_content(user_input_text, new_category)

if __name__ == "__main__":
    main()
