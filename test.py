import cv2
import pandas as pd
from ultralytics import YOLO  # Import YOLO class from your module
video = cv2.VideoCapture("https://toch-studio.s3.amazonaws.com/pravin/baseball/event/match_15/Match_15.csv_000204_000206.mp4")
model = YOLO("/home/multi-sy-003/Desktop/container_application_manager/auto_load_container/baseball/baseball_pitch_detection_container/KDeNYdtCE0IwRpv.pt")
coordinates = {}
counter = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    counter += 1
    if counter == 50:
        break
    result = model.predict(frame)
    frame_bbox_data = {'x1': [], 'x2': [], 'y1': [], 'y2': []}
    if result:
        for r in result:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()
                frame_bbox_data['x1'].append(x1)
                frame_bbox_data['x2'].append(x2)
                frame_bbox_data['y1'].append(y1)
                frame_bbox_data['y2'].append(y2)
    else:
        frame_bbox_data = {'x1': '', 'x2': '', 'y1': '', 'y2': ''}        #frame_bbox_data = {'x1': None, 'x2': None, 'y1': None, 'y2': None}
    coordinates[f"{counter}.jpg"] = frame_bbox_data
video.release()
# Convert the dictionary to DataFrame
data = pd.DataFrame.from_dict(coordinates, orient='index')
# Save DataFrame to CSV
data.to_csv("bbox_coordinates1.csv")
