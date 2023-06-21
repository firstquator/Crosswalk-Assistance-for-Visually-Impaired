import os
import cv2
import argparse
from FACE_RECOG import FACE_RECOG

# ./demo/crosswalk_02.mp4

parser = argparse.ArgumentParser(description="FACE_RECOGNITION")

parser.add_argument("--cam", type=bool, default=False, help="Use camera")
parser.add_argument("--webcam", type=bool, default=False, help="Use camera")
parser.add_argument("--video", type=str, default=None, help="Video file path")
parser.add_argument("--debug", type=bool, default=True, help="Debugging mode")
parser.add_argument("--size", type=tuple, default=(640, 640), help="Resize frame")
parser.add_argument("--save", type=str, default=None, help="Video save path")
parser.add_argument(
    "--save_name", type=str, default="video.mp4", help="Video save name"
)

if __name__ == "__main__":
    args = parser.parse_args()

    face = FACE_RECOG(args=args)
    face.start()
