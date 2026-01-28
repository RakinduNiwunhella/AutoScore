import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")  # nano = fast

# Open video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4)

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("HoopVision - Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()