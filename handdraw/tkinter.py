import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# 학습된 모델 불러오기
model = load_model('handwritten_digit_classifier.keras')

class HandwrittenDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Drawer")
        
        # Canvas 설정
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()

        # PIL 이미지 생성
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        # 마우스 이벤트 바인딩
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # 지우기 버튼
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # 예측 버튼
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.last_x, self.last_y = None, None

    def draw_digit(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=8)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # 이미지를 28x28로 리사이즈하고 모델에 입력
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0
        
        # CNN 모델로 예측
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        
        # 예측된 숫자 출력
        print(f"예측된 숫자는: {digit}")

# GUI 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwrittenDigitApp(root)
    root.mainloop()
