# LLaMA 모델을 학습시키기 위한 코드입니다. 전처리된 데이터를 로드하고, 학습 설정을 통해 모델을 학습합니다.

from transformers import LLaMAForCausalLM, LLaMATokenizer, Trainer, TrainingArguments

# 1. 모델 및 토크나이저 로드
tokenizer = LLaMATokenizer.from_pretrained("path_to_llama_tokenizer")
model = LLaMAForCausalLM.from_pretrained("path_to_llama_model")

# 2. 전처리된 데이터 불러오기 (텍스트 파일)
with open("path_to_preprocessed_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. 데이터 토큰화
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# 4. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",          # 결과 디렉토리
    num_train_epochs=3,              # 학습 epoch 수
    per_device_train_batch_size=2,   # 배치 크기
    per_device_eval_batch_size=2,    # 평가 배치 크기
    warmup_steps=500,                # 워밍업 스텝
    weight_decay=0.01,               # weight decay
    logging_dir="./logs",            # 로그 디렉토리
    logging_steps=10,
)

# 5. 학습 설정
trainer = Trainer(
    model=model,                          # 모델
    args=training_args,                   # 학습 인자
    train_dataset=inputs['input_ids'],    # 학습 데이터
)

# 6. 학습 시작
trainer.train()