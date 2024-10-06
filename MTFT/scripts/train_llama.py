from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import glob
from datasets import Dataset
import os
from dotenv import load_dotenv

# # .env 파일 로드
# load_dotenv()

# # 환경 변수에서 API 토큰 가져오기
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# # 1. 모델 및 토크나이저 로드 (API 토큰을 사용하여 인증)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_auth_token=HF_API_TOKEN)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_auth_token=HF_API_TOKEN)

# 0. Mac에서 MPS를 사용하는 경우 필수적인 설정
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# 1. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# pad_token 설정
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. 현재 시스템에 맞는 경로를 설정, 모든 .txt 파일 경로를 가져오기
base_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GitHub', 'sideProject', 'MTFT', 'DATA', 'preprocessed_data')
file_paths = glob.glob(os.path.join(base_path, '*.txt'))

print(file_paths)

# 2-2. 모든 파일의 텍스트를 불러와 하나의 리스트에 저장
all_text = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        all_text.extend(f.readlines())

# 3. 데이터 토큰화 (텍스트를 각 줄 단위로 나누어 인풋으로 변환)
inputs = tokenizer(all_text, return_tensors="pt", max_length=512, truncation=True, padding=True, add_special_tokens=True)

# Dataset에 'input_ids'와 'attention_mask' 추가
data = [{"input_ids": input_id, "attention_mask": attention_mask} 
        for input_id, attention_mask in zip(inputs['input_ids'], inputs['attention_mask'])]

# Dataset 생성
dataset = Dataset.from_list(data)

# 5. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",           # 결과 디렉토리
    num_train_epochs=3,               # 학습 epoch 수
    per_device_train_batch_size=1,    # 배치 크기
    per_device_eval_batch_size=1,     # 평가 배치 크기
    warmup_steps=500,                 # 워밍업 스텝
    weight_decay=0.01,                # weight decay
    logging_dir="./logs",             # 로그 디렉토리
    logging_steps=10,
    save_steps=1000,                  # 모델 저장 주기 (1000 스텝마다)
    save_total_limit=3,               # 저장할 체크포인트 최대 개수
    eval_strategy="steps",            # 평가 전략 설정 (스텝 단위로 평가)
    eval_steps=1000,                  # 평가 주기 (1000 스텝마다)
    fp16=True,                       # GPU 성능을 위해 16-bit 부동소수점 사용 (메모리 절약)
    no_cuda=True,
    gradient_checkpointing=False       # 메모리 절약  
)

# 6. Trainer에서 compute_loss를 오버라이드하여 loss를 계산
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 모델의 결과를 얻고 logits에서 loss 계산
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('input_ids')
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # CrossEntropyLoss 계산
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 7. 학습 설정
trainer = CustomTrainer(
    model=model,                       # 모델
    args=training_args,                # 학습 인자
    train_dataset=dataset,             # 학습 데이터셋
    eval_dataset=dataset,              # 평가 데이터셋 (학습과 동일 데이터로 설정 가능)
)

# 8. 학습 시작
trainer.train()

# 9. 학습이 완료된 후 모델 저장
save_path = os.path.join(base_path, 'saved_model')
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
