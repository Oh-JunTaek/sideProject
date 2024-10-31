from swarm import Swarm, Agent
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()  # .env 파일에서 환경 변수 로드
model = os.getenv('LLM_MODEL', 'llama3.1:8b')

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
client = Swarm(client=ollama_client)

# 에이전트 A: 계산기 역할
agent_a = Agent(
    name="Agent A",
    instructions="당신은 계산기입니다. 수학 문제에만 답변하세요.",
    model=model
)

# 에이전트 B: 과학 관련 질문 답변
agent_b = Agent(
    name="Agent B",
    instructions="과학 관련 질문에 대해서만 답변하세요.",
    model=model
)

# Agent C가 일반 응답을 생성하는 함수
def intelligent_route_message(message_content):
    # OpenAI API를 사용하여 메시지 내용을 분석하고 적절한 에이전트를 결정
    response = client.run(
        agent=Agent(
            name="Agent C",
            instructions="Analyze the user's intent and determine if the question is about mathematics or science."
        ),
        messages=[{"role": "user", "content": message_content}]
    )

    response_text = response.messages[-1]["content"].lower()

    if "math" in response_text or "calculation" in response_text:
        return agent_a
    elif "science" in response_text or "physics" in response_text or "chemistry" in response_text:
        return agent_b
    else:
        return "general"

# Agent C 정의: 기본적인 응답을 생성
agent_c = Agent(
    name="Agent C",
    instructions="해당 요청을 이해하지 못했습니다. 일반적인 질문에 대해 답변을 생성합니다.",
    model=model
)

# 대화형 루프 시작
while True:
    user_input = input("You: ")

    if user_input.lower() == "종료":
        print("Conversation ended.")
        break

    # Agent C가 사용자 요청을 분석하여 적절한 에이전트를 선택
    target_agent = intelligent_route_message(user_input)

    if target_agent == "general":
        # Agent C가 일반 응답 생성
        response = client.run(
            agent=agent_c,
            messages=[{"role": "user", "content": user_input}],
        )
        print("Agent C:", response.messages[-1]["content"])
    else:
        # 선택된 에이전트 실행
        response = client.run(
            agent=target_agent,
            messages=[{"role": "user", "content": user_input}],
        )
        print(f"{target_agent.name}:", response.messages[-1]["content"])
