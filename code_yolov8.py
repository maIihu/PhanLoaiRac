import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt


def load_model(model_path='yolov8n.pt'):
    """Load YOLOv8 model."""
    return YOLO(model_path)


def predict_image(image_path, model, conf_threshold=0.05, iou_threshold=0.5):
    """Run object detection on an image."""
    image = cv2.imread(image_path)
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    return results[0]


def save_cropped_objects(image_path, results, output_folder="assets/output"):
    """Save cropped objects detected by YOLO model."""
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)

    for idx, box in enumerate(results.boxes.data):
        x1, y1, x2, y2, conf, cls = map(int, box.tolist())
        cropped_object = image[y1:y2, x1:x2]
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_obj{idx}.jpg")
        cv2.imwrite(output_path, cropped_object)


def draw_boxes(image_path, results, class_map=None):
    """Draw bounding boxes on the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        label = f"{class_map[int(cls)] if class_map else int(cls)} ({conf:.2f})"

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def main(image_path, model_path='yolov8n.pt'):
    """Run detection, save cropped objects, and display the results."""
    model = load_model(model_path)
    results = predict_image(image_path, model)

    class_map = model.model.names if hasattr(model.model, 'names') else None

    save_cropped_objects(image_path, results)
    image_with_boxes = draw_boxes(image_path, results, class_map)

    plt.figure(figsize=(10, 6))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    image_path = "assets/input/6.png"
    main(image_path)