import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load model đã train
model = tf.keras.models.load_model("../asset/model.h5")

# Danh sách nhãn theo thứ tự model đã train
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Kích thước ảnh đầu vào
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """ Tiền xử lý ảnh trước khi đưa vào mô hình """
    img = cv2.imread(image_path)  # Đọc ảnh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    img = cv2.resize(img, IMG_SIZE)  # Resize về 224x224
    img = img / 255.0  # Chuẩn hóa giá trị pixel về [0,1]
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    return img


def predict_image(image_path):
    """ Dự đoán loại rác từ ảnh đầu vào """
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Dự đoán
    predicted_class = np.argmax(prediction)  # Lấy chỉ số của lớp có xác suất cao nhất
    confidence = np.max(prediction)  # Xác suất cao nhất

    print(f"Dự đoán: {class_labels[predicted_class]} ({confidence:.2%})")
    return class_labels[predicted_class]


def display_prediction(image_path):
    """ Hiển thị ảnh với nhãn dự đoán """
    predicted_label = predict_image(image_path)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()


# Dự đoán một ảnh
image_path = "../cardboard5.jpg"  # Thay đổi đường dẫn ảnh của bạn
display_prediction(image_path)