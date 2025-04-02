import shutil
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__, template_folder='../templates')

# Tải mô hình phân loại rác
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = tf.keras.models.load_model("../res/model.h5")

# Tải mô hình YOLOv8
yolo_model = YOLO("../res/yolov8n.pt")

# Thư mục lưu ảnh cắt
output_folder = 'static/assets/output'
os.makedirs(output_folder, exist_ok=True)

# Dự đoán bằng YOLOv8
def predict_image(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image, conf=0.05, iou=0.5, verbose=True)[0]  # Giảm conf để nhận diện nhiều vật thể hơn
    print(f"Detected {len(results.boxes)} objects")  # Log số lượng vật thể phát hiện được
    return results

# Lưu vật thể cắt ra từ ảnh
def save_cropped_objects(image_path, results):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)

    saved_files = []
    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_object = image[y1:y2, x1:x2]
        if cropped_object.size == 0:
            continue  # Bỏ qua nếu cắt lỗi
        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_obj{idx}.jpg")
        cv2.imwrite(output_path, cropped_object)
        saved_files.append(output_path)

    return saved_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_output_folder', methods=['GET'])
def clear_output_folder():
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Output folder not found"})
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Lưu ảnh mới
    image_path = os.path.join(output_folder, 'uploaded_image.jpg')
    image.save(image_path)

    # Dự đoán với YOLO
    results = predict_image(image_path)
    cropped_images = save_cropped_objects(image_path, results)

    # Chuyển đường dẫn thành URL hợp lệ
    cropped_images = [f"/static/assets/output/{os.path.basename(img)}" for img in cropped_images]

    if not cropped_images:
        return jsonify({"error": "No objects detected"}), 400

    return jsonify({"message": "Image processed", "cropped_images": cropped_images})

# API Phân loại ảnh
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")

    # Xử lý ảnh
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Dự đoán
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx] * 100)

    return jsonify({"predicted_class": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
