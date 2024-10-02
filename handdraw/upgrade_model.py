import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 기존 학습된 모델 불러오기
model = tf.keras.models.load_model('./model/handwritten_digit_classifier.keras')

# 데이터 증강을 위한 설정
datagen = ImageDataGenerator(
    rotation_range=5,    # 회전 범위를 더 작게
    zoom_range=0.05,     # 확대/축소 범위 축소
    width_shift_range=0.05,  # 가로 이동 범위 축소
    height_shift_range=0.05  # 세로 이동 범위 축소
)

# MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리 (정규화 및 차원 추가)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 숫자 8의 빈도를 랜덤하게 10~20% 줄이기
eight_indices = np.where(train_labels == 8)[0]
reduce_count = int(len(eight_indices) * np.random.uniform(0.1, 0.2))  # 10%~20% 줄임
remove_indices = np.random.choice(eight_indices, size=reduce_count, replace=False)

# 8의 일부를 제거하여 새로운 데이터셋 생성
train_images = np.delete(train_images, remove_indices, axis=0)
train_labels = np.delete(train_labels, remove_indices, axis=0)

# 0, 6, 9에 대해 강화된 데이터 증강 적용
for idx in range(len(train_labels)):
    if train_labels[idx] in [0, 6, 9]:  # 0, 6, 9에 대해서만 증강
        train_images[idx] = datagen.random_transform(train_images[idx].reshape(28, 28, 1))

# 데이터 증강을 통한 추가 학습
train_generator = datagen.flow(train_images, train_labels, batch_size=64)

# 모델 학습 (증강된 데이터를 사용하여 추가 학습)
model.fit(train_generator, epochs=5, validation_data=(test_images, test_labels))

# 학습된 모델 저장
model.save('handwritten_digit_classifier_augmented3.keras')
