from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = YOLO('yolov8n.pt')

# Load image
image_path = 'assets/input/im1.png'  # Thay bằng đường dẫn ảnh của bạn
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model(image)

# Plot results
for result in results:
    for box in result.boxes.xyxy:  # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật

plt.imshow(image)
plt.axis('off')
plt.show()