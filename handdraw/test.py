import numpy as np
from tensorflow.keras.datasets import mnist

# MNIST 데이터셋 불러오기
(train_images, train_labels), _ = mnist.load_data()

# 각 숫자 라벨의 개수 세기
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))
