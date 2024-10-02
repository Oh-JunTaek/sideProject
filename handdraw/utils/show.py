import matplotlib.pyplot as plt
import tensorflow as tf

# MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 첫 10개의 훈련 이미지와 라벨 확인
for i in range(10):
    plt.subplot(2, 5, i+1)  # 2행 5열의 플롯
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')

plt.show()
