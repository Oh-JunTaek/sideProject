from transformers import LLaMAForCausalLM, LLaMATokenizer, Trainer, TrainingArguments
import torch
import glob
from datasets import Dataset

# 1. 모델 및 토크나이저 로드
tokenizer = LLaMATokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = LLaMAForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. 모든 전처리된 텍스트 파일을 불러오기 (경로에 있는 .txt 파일 모두)
file_paths = glob.glob(r"C:\Users\dev\Documents\GitHub\sideProject\MTFT\DATA\preprocessed_data\*.txt")

# 2-2. 모든 파일의 텍스트를 불러와 하나의 리스트에 저장
all_text = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        all_text.extend(f.readlines())

# 3. 데이터 토큰화 (텍스트를 인풋으로 변환)
inputs = tokenizer(all_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

# 4. Hugging Face의 Dataset으로 변환
# Dataset에 'input_ids'와 'attention_mask' 추가
data = [{"input_ids": input_id, "attention_mask": attention_mask} 
        for input_id, attention_mask in zip(inputs['input_ids'], inputs['attention_mask'])]

# Dataset 생성
dataset = Dataset.from_list(data)

# 5. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",           # 결과 디렉토리
    num_train_epochs=3,               # 학습 epoch 수
    per_device_train_batch_size=2,    # 배치 크기
    per_device_eval_batch_size=2,     # 평가 배치 크기
    warmup_steps=500,                 # 워밍업 스텝
    weight_decay=0.01,                # weight decay
    logging_dir="./logs",             # 로그 디렉토리
    logging_steps=10,
    save_steps=1000,                  # 모델 저장 주기 (1000 스텝마다)
    save_total_limit=3,               # 저장할 체크포인트 최대 개수
    evaluation_strategy="steps",      # 평가 전략 설정 (스텝 단위로 평가)
    eval_steps=1000,                  # 평가 주기 (1000 스텝마다)
    fp16=True                         # GPU 성능을 위해 16-bit 부동소수점 사용 (메모리 절약)
)

# 6. 학습 설정
trainer = Trainer(
    model=model,                       # 모델
    args=training_args,                # 학습 인자
    train_dataset=dataset,             # 학습 데이터셋
    eval_dataset=dataset,              # 평가 데이터셋 (학습과 동일 데이터로 설정 가능)
)

# 7. 학습 시작
trainer.train()

# 8. 학습이 완료된 후 모델 저장
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")
