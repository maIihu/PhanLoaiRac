import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("model.h5")

# Danh sách nhãn (theo thứ tự khi huấn luyện)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def predict_trash(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không thể đọc ảnh. Hãy kiểm tra đường dẫn!")
        return

    # Tiền xử lý ảnh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    img = cv2.resize(img, (224, 224))  # Resize về kích thước phù hợp
    img = img / 255.0  # Chuẩn hóa về khoảng [0,1]
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Dự đoán với mô hình
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)  # Lấy nhãn có xác suất cao nhất
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100  # Xác suất %

    # Hiển thị kết quả
    print(f"🗑️ Loại rác dự đoán: {predicted_class} ({confidence:.2f}%)")
    cv2.imshow(f"Prediction: {predicted_class} ({confidence:.2f}%)", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_image():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(title="Chọn ảnh rác", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        predict_trash(file_path)


if __name__ == "__main__":
    select_image()
