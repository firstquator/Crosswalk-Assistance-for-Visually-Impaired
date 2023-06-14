import os
import cv2
import argparse
import numpy as np
from CROSSWALK_V1 import HELP_CROSSWALK

# ./demo/crosswalk_02.mp4

parser = argparse.ArgumentParser(description="HELP_CROSSWALK")

parser.add_argument(
    "--mode",
    type=int,
    default=3,
    help="0: find zebra crossing, 1: location pedistrain, 2: detect traffic light, 3: all",
)
parser.add_argument("--cam", type=bool, default=False, help="Use camera")
parser.add_argument("--webcam", type=bool, default=False, help="Use camera")
parser.add_argument("--video", type=str, default=None, help="Video file path")
parser.add_argument("--size", type=tuple, default=(640, 640), help="Resize frame")
parser.add_argument("--save", type=str, default=None, help="Video save path")
parser.add_argument(
    "--save_name", type=str, default="video.mp4", help="Video save name"
)
parser.add_argument("--fps", type=float, default=20.0, help="Video FPS")
parser.add_argument(
    "--onnx", type=str, default="./YOLO/models/crosswalk_n.onnx", help="Onnx path"
)
parser.add_argument("--debug", type=bool, default=False, help="Debugging mode")
parser.add_argument("--voice", type=bool, default=True, help="Voice control")
parser.add_argument("--demo", type=bool, default=False, help="Debugging mode")


args = parser.parse_args()

if args.cam:
    key = "/dev/video0"
elif args.webcam:
    key = 0
elif args.video:
    key = args.video
elif args.demo:
    key = "./Demo/crosswalk_02.mp4"

cap = cv2.VideoCapture(key)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
crosswalk = HELP_CROSSWALK(debug=args.debug, voice=args.voice)
crosswalk.set_onnx(onnx_path=args.onnx)

if args.save:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"{os.path.join(args.save, args.save_name)}", fourcc, args.fps, args.size
    )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, args.size)
    frame = crosswalk(frame, mode=args.mode)
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
