from tensorflow.keras.models import load_model

# 기존 .h5 형식 모델 불러오기
model = load_model('handwritten_digit_classifier.h5')

# 새로운 Keras 형식으로 저장
model.save('handwritten_digit_classifier.keras')