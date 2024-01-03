from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
video_path = './test1.mp4'
video = cv2.VideoCapture(video_path)
frames_available = True
while frames_available:
    frames_available, frame = video.read()

    if frames_available:
        results = model.track(frame, persist=True) 
        frame_with_boxes = results[0].plot() 
        cv2.imshow('frame',frame_with_boxes)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break







