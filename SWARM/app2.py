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

def transfer_to_agent_b():
    print("Agent A가 작업을 Agent B에게 전달합니다.")
    return agent_b


agent_a = Agent(
    name="Agent A",
    instructions="너는 영어 번역기야. 입력받은 메시지를 영어로 반환해",
    functions=[transfer_to_agent_b],
    model=model
)

agent_b = Agent(
    name="Agent B",
    instructions="너는 단어 추출기야. 입력받은 텍스트에서 핵심이라고 생각되는 단어 1개만 출력해",
    model=model
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "이 문장을 영어로 번역하고 중요한 단어 하나를 추출해줘: 지구는 태양계의 세 번째 행성입니다."}],
)

print(response.messages[-1]["content"])