from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 증강을 위한 설정
datagen = ImageDataGenerator(
    rotation_range=10,   # 10도 회전
    zoom_range=0.1,      # 10% 확대/축소
    width_shift_range=0.1,  # 가로로 10% 이동
    height_shift_range=0.1  # 세로로 10% 이동
)
