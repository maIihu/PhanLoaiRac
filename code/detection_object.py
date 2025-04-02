import cv2
import os
from ultralytics import YOLO


def load_model(model_path='../res/yolov8n.pt'):
    return YOLO(model_path)


def predict_image(image_path, model):
    image = cv2.imread(image_path)
    return model(image, conf=0.05, iou=0.5, verbose=False)[0]


def save_cropped_objects(image_path, results, output_folder='../static/assets/output'):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)

    saved_files = []
    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_object = image[y1:y2, x1:x2]
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_obj{idx}.jpg")
        cv2.imwrite(output_path, cropped_object)
        saved_files.append(output_path)

    if saved_files:
        print(f"Successfully detected {len(saved_files)} objects:")
        for file in saved_files:
            print(f" - {file}")
    else:
        print("No results")


def main(image_path, model_path='../res/yolov8n.pt'):
    model = load_model(model_path)
    results = predict_image(image_path, model)
    save_cropped_objects(image_path, results)
    print("Processing detected")


if __name__ == "__main__":
    main("../assets/input/6.png")
