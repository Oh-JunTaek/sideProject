from llama import get_llama_model

class Completions:
    @staticmethod
    def create(model, messages, **kwargs):
        # LLaMA 모델을 호출하여 응답을 생성
        prompt_text = messages[-1]["content"]
        response_text = get_llama_model(prompt_text)
        return {"choices": [{"message": {"content": response_text}}]}  # Swarm에서 필요한 형태로 반환

class LLaMAClient:
    def __init__(self):
        self.chat = Completions()  # chat 속성으로 Completions 인스턴스를 설정
