from swarm import Swarm, Agent
from swarm.repl import run_demo_loop 

client = Swarm()

def transfer_to_agent_b():
    return agent_b

def traansfer_to_agent_c():
    return agent_c


agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",

    functions=[transfer_to_agent_b,traansfer_to_agent_c],
)

agent_b = Agent(
    name="Agent B",
    instructions="입력받은 값을 일본어로 바꿔서 답변해줘.",
)

agent_c = Agent(
    name="Ally",
    instructions="너는 입력받은 값을 바탕으로 새로운 질문을 생성해 줘"
)

# response = client.run(
#     agent=agent_a,
#     messages=[{"role": "user", "content": "I want to talk to ally. 지구는 은하계에서 몇번째 행성이야?"}],
# )

run_demo_loop(agent_a, stream=True)
# print(response.messages[-1]["content"])