import tensorflow as tf

# 학습된 모델 불러오기
model = tf.keras.models.load_model('./model/handwritten_digit_classifier_augmented.keras')

# MNIST 테스트 데이터셋 불러오기
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리 (정규화 및 차원 추가)
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 모델 성능 평가
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"테스트 정확도: {accuracy * 100:.2f}%")
