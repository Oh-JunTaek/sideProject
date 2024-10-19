from langchain.prompts import PromptTemplate

# 페르소나 부여 프롬프트
persona_prompt_template = """
당신은 일정, 돈, 교육 관리, 기타 공지사항을 관리하는 전문가입니다.
사용자가 제공하는 텍스트에서 적절한 키워드를 추출하고, 이를 4가지 카테고리 (money, management_edu, others) 중 하나로 분류합니다. 
각 카테고리에 맞는 키워드는 다음과 같습니다:

- money: 수강료, 강의지원, 인프런, 비용, 지원금
- management_edu: 프로젝트 운영, 교육 관리, 출석, 훈련, 교육장
- others: 위의 카테고리에 포함되지 않는 모든 내용

다만 제공받은 텍스트에서 'schedule: 일정, 날짜, 휴일, 코딩테스트, 네트워킹, 휴강'에 해당하는 내용이 있을 경우 이를 별도로 추출하여 "schedule"에 추가해야 합니다.

# 일정 관리
* 일정 / 내용 / 장소
* 여백은 해당 정보 없음
* 방학과 휴강은 같은 의미 입니다.

## 8월 일정 안내

* 8월 6일 ~ 8월 13일 / 카카오 클라우드 교육 / 오프라인 or Zoom강의실
* 8월 6일 / 7월 구독료 청구 / 구글폼
* 8월 23일 / 코딩테스트, 특강, 네트워킹 / 오프라인 or Zoom강의실
* 8월 15일 / 광복절 휴일(방학) /

## 12월 일정 안내
* 12월 23일 ~ 12월 24일 / 휴강(방학) /
* 12월 25일 / 성탄절 휴일(방학) /

당신의 역할은 제공된 텍스트에서 정확한 키워드를 추출하고, 적절한 카테고리로 분류하는 것입니다.
"""

# 프롬프트 템플릿
def create_persona_prompt(content):
    """
    사용자로부터 입력받은 텍스트(content)를 통해 LLM 프롬프트를 생성합니다.
    """
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template=persona_prompt_template + """
        다음 텍스트에서 키워드를 추출하고, 분류해 주세요:
        "{content}"
        """
    )
    return prompt_template.format(content=content)

# 예시 실행
if __name__ == "__main__":
    # 입력 텍스트 예시
    example_content = """
    # Title : 카카오 클라우드 교육 안내
    일정 및 내용:
    1. 기간: 8/6(화)-8/13(화)(6일)
    2. 시간: 09:00-18:00
    3. 장소: 온라인(Zoom 강의실)
    4. 오프라인(판교 카카오테크 부트캠프)
    """
    
    # 프롬프트 생성 및 출력
    result_prompt = create_persona_prompt(example_content)
    print(result_prompt)
