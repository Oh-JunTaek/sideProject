from swarm import Swarm, Agent
from dotenv import load_dotenv
from swarm.repl import run_demo_loop 
import os
from utils.weather import weather_agent,get_weather
from utils.rag import rag_agent,generate_response

load_dotenv()


client = Swarm()

def transfer_to_agent_a():
    return agent_a

def transfer_to_agent_b():
    return agent_b

def transfer_to_weather_agent():
    return weather_agent

def transfer_to_rag_agent():
    return rag_agent

agent_a = Agent(
    name="비서",
    instructions="사용자의 요청을 분석하고, 적절한 에이전트에게 전달하세요.적절한 agent가 없다면 당신이 답변하세요.",
    functions=[transfer_to_agent_b, transfer_to_weather_agent,transfer_to_rag_agent],
)

agent_b = Agent(
    name="번역가",
    instructions="당신은 번역가 입니다. 주어진 정보를 번역하는 역할입니다.",
    functions=[transfer_to_agent_a],
    stream=True,
)

weather_agent = Agent(
    name="Weather Agent",
    instructions="당신은 기상캐스터입니다. 날씨정보를 제공할때 참고한 시간대를 함께 알려주세요.",
    functions=[get_weather,transfer_to_agent_a],
)

# RAG 에이전트 설정
rag_agent = Agent(
    name="RAG Agent",
    instructions="사용자의 질문에 대해 검색된 정보를 기반으로 답변을 생성해 주세요.",
    functions=[generate_response,transfer_to_agent_a],
)




run_demo_loop(agent_a, stream=True)  # stream=True를 통해 실시간 평가 모드
# print(response.messages[-1]["content"])