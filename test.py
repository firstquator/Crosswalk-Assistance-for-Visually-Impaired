import cv2
import numpy as np
from CROSSWALK import CROSSWALK

test1 = "C:/Projects/help_crosswalk/Crosswalk-Recognition/processedVideo.mp4"
test2 = "./Datasets/Demo_videos/crosswalk_03.mp4"

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./Datasets/Demo_videos/crosswalk_03.mp4', fourcc, 20.0, (640, 640))

cap = cv2.VideoCapture(test2)
crosswalk = CROSSWALK()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    k, frame = crosswalk(frame)
    cv2.imshow("Detected Images", frame)
    # out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord(" "):  # 스페이스바를 누르면 일시정지/재생
        cv2.waitKey(0)  # 아무 키나 누를 때까지 기다림

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
# out.release()
