import os
import cv2
import argparse
from FACE_RECOG import FACE_RECOG

# ./demo/crosswalk_02.mp4

parser = argparse.ArgumentParser(description="FACE_RECOGNITION")

parser.add_argument("--cam", type=bool, default=False, help="Use camera")
parser.add_argument("--webcam", type=bool, default=False, help="Use camera")
parser.add_argument("--video", type=str, default=None, help="Video file path")
parser.add_argument("--debug", type=bool, default=False, help="Debugging mode")
parser.add_argument("--size", type=tuple, default=(640, 640), help="Resize frame")
parser.add_argument("--save", type=str, default=None, help="Video save path")
parser.add_argument(
    "--save_name", type=str, default="video.mp4", help="Video save name"
)

args = parser.parse_args()

if args.cam:
    key = "/dev/video0"
    cap = cv2.VideoCapture(key, cv2.CAP_V4L2)
elif args.webcam:
    key = 0
    cap = cv2.VideoCapture(key)
elif args.video:
    key = args.video
    cap = cv2.VideoCapture(key)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
face = FACE_RECOG(debug=args.debug)

if args.save:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"{os.path.join(args.save, args.save_name)}", fourcc, args.fps, args.size
    )

cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    button = cv2.waitKey(1)

    frame = cv2.resize(frame, args.size)
    frame = face(frame, cnt, key=button)
    cv2.imshow("Detected Frame", frame)

    if args.save:
        out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord(" "):  # Spacebar to pause/play
        cv2.waitKey(0)  # Wait for any key to be pressed

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
if args.save:
    out.release()
