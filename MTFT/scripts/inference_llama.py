# 학습된 LLaMA 모델로부터 텍스트를 생성하거나 추론할 수 있는 코드입니다. 학습된 모델을 불러와서 텍스트 생성을 수행합니다.

from transformers import LLaMAForCausalLM, LLaMATokenizer

# 학습된 모델과 토크나이저 로드
model = LLaMAForCausalLM.from_pretrained("path_to_saved_model")
tokenizer = LLaMATokenizer.from_pretrained("path_to_saved_tokenizer")

# 예시 텍스트
input_text = "에너지 단위의 차이는 무엇인가요?"

# 토큰화
inputs = tokenizer(input_text, return_tensors="pt")

# 텍스트 생성
output = model.generate(inputs["input_ids"])

# 결과 출력
print(tokenizer.decode(output[0], skip_special_tokens=True))