from ultralytics import YOLO
import cv2
import torch


def detect_objects(image_path, model_path='yolov8n.pt'):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image. Check the path.")

    # Perform object detection
    results = model(image)

    # Extract detected objects
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detected_objects.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display the image with detections
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output saved at {output_path}")

    return detected_objects


# Example usage
if __name__ == "__main__":
    image_path = "assets/input/6.png"  # Đường dẫn ảnh đầu vào
    objects = detect_objects(image_path)
    print(objects)
