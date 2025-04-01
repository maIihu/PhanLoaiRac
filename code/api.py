from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model = tf.keras.models.load_model("res/model.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(BytesIO(file.read()))
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx] * 100)

    return jsonify({"predicted_class": predicted_class, "confidence": confidence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
