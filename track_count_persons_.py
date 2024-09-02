import ultralytics
import torch
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import tensorflow as tf
# Define class names for detection
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# Load YOLOv8 model
model_path = Path("yolov8m.pt")
model = YOLO(model_path)
# DeepSORT configuration
deep_sort_weights = Path('deep_sort/deep/checkpoint/ckpt.t7')
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
# Load Gender Classification Model
gender_model_path = Path('gender_classification_vgg19.h5')
gender_model = tf.keras.models.load_model(gender_model_path)
# Define the video path
video_path = Path('/home/hermen/Pictures/yoloGD/People In Restaurant - Free HD Stock Video Footage - No Copyright - Shoppingmall Bar Cafe.mp4')
cap = cv2.VideoCapture(str(video_path))
# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = Path('output.mp4')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Frame interval (frames to skip for every 5 seconds)
frame_interval = int(fps * 5)
frames = []
unique_track_ids = set()
counter, elapsed = 0, 0
start_time = time.perf_counter()
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
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()
        results = model(frame, device=device, classes=0, conf=0.4)
        for result in results:
            boxes = result.boxes
            cls = boxes.cls.tolist()
            conf = boxes.conf
            xyxy = boxes.xyxy
            xywh = boxes.xywh
            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh.detach().cpu().numpy()
            tracks = tracker.update(bboxes_xywh, conf, og_frame)
            for track in tracker.tracker.tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_tlbr()
                w = x2 - x1
                h = y2 - y1
                red_color = (0, 0, 255)
                blue_color = (255, 0, 0)
                green_color = (0, 255, 0)
                color_id = track_id % 3
                color = red_color if color_id == 0 else blue_color if color_id == 1 else green_color
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                if cls:
                    class_index = int(cls[0])
                    class_name = class_names[class_index] if 0 <= class_index < len(class_names) else "Unknown"
                else:
                    class_name = "Unknown"
                cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # Use entire bounding box for gender prediction
                box_image = og_frame[int(y1):int(y2), int(x1):int(x2)]
                if box_image.size > 0:
                    gender = classify_gender(box_image)
                    cv2.putText(og_frame, f"Gender: {gender}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(og_frame, "Gender: Unknown", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                unique_track_ids.add(track_id)
        person_count = len(unique_track_ids)
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time
        cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(og_frame)
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow("Video", og_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1
cap.release()
out.release()
cv2.destroyAllWindows()