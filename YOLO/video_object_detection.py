import cv2
from YOLO import YOLO

# Initialize the webcam
cap = cv2.VideoCapture("../Datasets/Demo_videos/night_02.mp4")

# Initialize YOLO object detector
model_path = "./models/crosswalk_n.onnx"
yolo_detector = YOLO(model_path, conf_thres=0.22, iou_thres=0.3)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('../Datasets/videos/crosswalk_03_m.mp4', fourcc, 20.0, (1080, 960))

cnt = 1
while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))
    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolo_detector(frame)

    combined_img = yolo_detector.draw_detections(frame)
    # cv2.imwrite(f'C:/Projects/help_crosswalk/Datasets/frames/{str(cnt).zfill(5)}.jpg', combined_img)    # Save frame

    cnt += 1
    # out.write(combined_img)

    cv2.imshow("Detected Images", combined_img)

    if len(boxes) == 0:
        continue

    # for box, score in zip(boxes, scores):
    #     if score < 0.22: continue
    #     box[box < 0] = 0
    #     x1, y1, x2 ,y2 = box.astype('int')
    #     cv2.imshow('Clip', frame[y1:y2, x1:x2, :])

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# out.release()
