import re

# 각 카테고리별 파일 경로 정의
file_paths = {
    "money": "data/money.md",
    "management_edu": "data/management_edu.md",
    "schedule": "data/schedule.md",
    "others": "data/others.md"
}

# 텍스트에서 날짜를 추출하는 함수 (일정 업데이트를 위해 사용)
def extract_date(text):
    match = re.search(r'(\d{1,2})월\s*(\d{1,2})일', text)
    if match:
        return f"{match.group(1)}월 {match.group(2)}일"
    return None

# schedule.md 파일에서 동일한 일정이 이미 존재하는지 확인하는 함수
def schedule_exists(schedule_entry, schedule_file):
    with open(schedule_file, 'r', encoding='utf-8') as file:
        content = file.read()
        return schedule_entry in content

# 카테고리별로 적절한 파일에 내용을 추가하는 함수 (일정 포함)
def append_to_file(category, content, is_schedule=False):
    file_path = file_paths[category]  # 카테고리별 파일 경로 선택
    
    if is_schedule:
        # 일정 관련 내용을 추출하고 포맷을 정리
        date = extract_date(content)
        place = "Zoom 강의실 또는 오프라인" if "Zoom" in content or "오프라인" in content else "장소 없음"
        schedule_entry = f"{date} / {content[:30]} / {place}"
        
        # 동일한 일정이 이미 존재하는지 확인 후, 없으면 추가
        if not schedule_exists(schedule_entry, file_paths["schedule"]):
            with open(file_paths["schedule"], 'a', encoding='utf-8') as schedule_file:
                schedule_file.write(f"* {schedule_entry}\n")
    else:
        # 카테고리 파일에 해당 내용을 추가
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"\n# {content.strip()}\n")

# 텍스트를 처리하고 분류된 카테고리에 맞게 파일에 내용을 추가하는 함수
def handle_content(text, category):
    """
    텍스트를 카테고리별로 분류한 후, 해당 카테고리의 .md 파일에 내용을 추가.
    일정이 포함된 경우, schedule.md에도 추가.
    """
    append_to_file(category, text)
    
    # 텍스트에 일정 관련 정보가 포함된 경우, schedule.md에도 추가
    if "일정" in text or extract_date(text):
        append_to_file("schedule", text, is_schedule=True)

# 사용자에게 분류 및 일정 추가 작업을 확인하는 함수
def confirm_and_process(text, category, detected_schedule=None):
    """
    사용자에게 분류 및 일정 정보를 확인한 후, y/n 입력에 따라 처리.
    """
    if detected_schedule:
        print(f"해당 공지 내용을 {category}로 분류했고, 일정도 포함되어 있어서 {detected_schedule}으로 분류했는데 맞습니까? (y/n)")
    else:
        print(f"해당 공지 내용을 {category}로 분류했는데 맞습니까? (y/n)")
    
    user_input = input().lower()
    
    if user_input == "y":
        # 사용자가 동의하면 파일에 추가
        handle_content(text, category)
        print("내용이 성공적으로 추가되었습니다.")
    else:
        # 재분류 및 날짜 정보 수정을 요청
        print("해당 공지의 카테고리를 숫자로 입력하세요. 1. money 2. management_edu 3. others")
        category_input = input()
        category_mapping = {
            "1": "money",
            "2": "management_edu",
            "3": "others"
        }
        new_category = category_mapping.get(category_input, "others")
        
        print("해당 공지의 날짜 및 내용을 스페이스바를 기준으로 알려주세요 (예시: 8월13일 코딩테스트 장소없음)")
        schedule_input = input()

        # 새로운 정보로 처리
        handle_content(text, new_category)
        
        # 일정이 있으면 일정에도 추가 (장소 입력이 없는 경우 기본값 "장소 없음"으로 처리)
        if schedule_input:
            # 스페이스바를 기준으로 나눈 입력값 처리
            schedule_parts = schedule_input.split(" ", 1)
            
            if len(schedule_parts) == 2:
                date, description = schedule_parts
                place = description.split(" ", 1)[1] if len(description.split(" ", 1)) > 1 else "장소 없음"
            else:
                date = schedule_parts[0]
                place = "장소 없음"
            
            schedule_entry = f"{date} / {schedule_parts[0]} / {place}"
            append_to_file("schedule", schedule_entry, is_schedule=True)

        print("내용이 수정되어 성공적으로 추가되었습니다.")
