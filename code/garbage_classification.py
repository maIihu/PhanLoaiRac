import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("../res/model.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def predict_trash(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224)) / 255.0
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)] * 100

    print(f"{os.path.basename(image_path)} → {predicted_class} ({confidence:.2f}%)")
    return predicted_class, confidence


def process_output_images(output_folder="../assets/output"):
    for img_file in os.listdir(output_folder):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            predict_trash(os.path.join(output_folder, img_file))


if __name__ == "__main__":
    process_output_images()
