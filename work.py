import ultralytics
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import tensorflow as tf

# Define class names for detection
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load YOLOv8 model
model_path = "yolov8m.pt"
model = YOLO(model_path)

# Load DeepSORT model
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Load Gender Classification Model
gender_model_path = 'gender_classification_vgg19.h5'
gender_model = tf.keras.models.load_model(gender_model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image):
    # Resize to model's expected input size (adjust size based on your model requirements)
    image = cv2.resize(image, (100, 100))
    image = image.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify_gender(image):
    preprocessed_image = preprocess_image(image)
    prediction = gender_model.predict(preprocessed_image)
    gender = 'Male' if np.argmax(prediction) == 0 else 'Female'
    return gender

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()

    # Perform YOLOv8 object detection
    results = model(frame, device=device, classes=0, conf=0.4)

    for result in results:
        boxes = result.boxes
        cls = boxes.cls.tolist()
        xyxy = boxes.xyxy
        xywh = boxes.xywh
        confidences = boxes.conf.cpu().numpy()  # Extract confidence scores

        pred_cls = np.array(cls)
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh.detach().cpu().numpy()

        tracks = tracker.update(bboxes_xywh, confidences, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()

            # Draw bounding boxes
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Get class name
            if cls:
                class_index = int(cls[0])
                class_name = class_names[class_index] if 0 <= class_index < len(class_names) else "Unknown"
            else:
                class_name = "Unknown"

            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Extract face for gender classification
            box_image = og_frame[int(y1):int(y2), int(x1):int(x2)]
            if box_image.size > 0:
                gender = classify_gender(box_image)
                cv2.putText(og_frame, f"Gender: {gender}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(og_frame, "Gender: Unknown", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the frame with predictions
    cv2.imshow("Webcam Feed", cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
