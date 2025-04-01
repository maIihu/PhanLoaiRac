import tensorflow as tf
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("res/model.h5")

# Danh sách nhãn (theo thứ tự khi huấn luyện)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def predict_trash(image_path):
    """Dự đoán loại rác từ ảnh đầu vào."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    print(f"🗑️ {os.path.basename(image_path)} → {predicted_class} ({confidence:.2f}%)")
    return predicted_class, confidence


def process_output_images(output_folder="assets/output"):
    """Duyệt tất cả ảnh trong thư mục output và dự đoán kết quả."""
    output_images = [f for f in os.listdir(output_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

    for img_file in output_images:
        img_path = os.path.join(output_folder, img_file)
        predict_trash(img_path)


def select_image():
    """Mở hộp thoại chọn ảnh và thực hiện dự đoán."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Chọn ảnh rác", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        predict_trash(file_path)


if __name__ == "__main__":
    process_output_images()
