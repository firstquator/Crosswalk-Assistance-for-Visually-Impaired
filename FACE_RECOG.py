import os
import cv2
import time
import copy
import threading
import numpy as np
import face_recognition as face

from gtts import gTTS
from pathlib import Path
from collections import OrderedDict


class FACE_RECOG:
    CONFIG = {
        "database_dir": "./FACE_LIST/",
        "sound_dir": "./Sound/face_recog",
        "capture_size": {"fx": 1.0, "fy": 1.0},
        "recog_size": {"fx": 0.25, "fy": 0.25},
        "distance_thres": 0.3,
        "process_frame": 1,
        "check_face": 5,
        "check_no_face": 35,
        "voice_interval": 1.2,
        "voice_delay": 3,
    }

    CHECK = {}
    THREAD = {"voice": 1, "register": 0, "recog": 0}
    LOCATION = 0
    NO_FACE = 0

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.encoding = {}
        self.frame = None
        self.frame_num = 0
        self.key = -1
        self.regist = False
        self.recognized_face = []

        self.__loading()

    def start(self):
        streaming = self.__set_thread(
            target=self.__streaming, kwargs_dict={"args": self.args}
        )
        recognition = self.__set_thread(target=self.__face_recognition)

        streaming.start()
        recognition.start()

        recognition.join()
        streaming.join()

        cv2.destroyAllWindows()

    def set_config(self, param, kwards):
        self.CONFIG[param] = kwards

    ########################## Private Functions ##########################
    def __streaming(self, args):
        if args.cam:
            key = "/dev/video0"
            cap = cv2.VideoCapture(key, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif args.webcam:
            key = 0
            cap = cv2.VideoCapture(key)
        elif args.video:
            key = args.video
            cap = cv2.VideoCapture(key)

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                f"{os.path.join(args.save, args.save_name)}",
                fourcc,
                args.fps,
                args.size,
            )

        while cap.isOpened():
            ret, self.frame = cap.read()
            self.key = cv2.waitKey(25)
            if self.debug:
                for name, bbox in self.recognized_face:
                    # Rescaling
                    bbox = (
                        np.array(bbox) * 1 / self.CONFIG["recog_size"]["fx"]
                    ).astype(np.int32)
                    # Draw bbox
                    top, right, bottom, left = bbox
                    cv2.rectangle(
                        self.frame, (left, top), (right, bottom), (0, 0, 255), 2
                    )
                    cv2.rectangle(
                        self.frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        self.frame,
                        name,
                        (left + 6, bottom + 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 255, 255),
                        1,
                    )

            cv2.imshow("Face Recognition", self.frame)

            if self.key == ord("q"):
                cv2.destroyAllWindows()
                break

    def __face_recognition(self):
        while True:
            if self.key == ord("q"):
                break

            if self.key == ord("r"):
                self.__register()

            if self.THREAD["recog"] == 0:
                self.__recognition()

            self.frame_num += 1
            if self.frame_num > self.CONFIG["process_frame"]:
                self.frame_num = 0
            time.sleep(0.01)

        print("Finish Face Recognition")

    def __register(self):
        image = self.frame.copy()
        image_capture = cv2.resize(
            image,
            (0, 0),
            fx=self.CONFIG["capture_size"]["fx"],
            fy=self.CONFIG["capture_size"]["fy"],
        )

        # Check Error : No face or multiple faces
        face_locations = face.face_locations(image_capture)
        if len(face_locations) != 1:
            if self.debug:
                print(
                    f"I found {len(face_locations)} face(s) in this photograph. so capture canceled."
                )
            return

        # Input name
        name = input("Please Enter a name : ")

        # Save images
        save_path = os.path.join(self.CONFIG["database_dir"], name + ".jpg")
        cv2.imwrite(save_path, image)

        # Save name voice
        sound_path = os.path.join(self.CONFIG["sound_dir"], name + ".mp3")

        tts = gTTS(name, lang="en")
        tts.save(sound_path)

        # Encoding New face
        self.encoding[name] = face.face_encodings(image, face_locations)[0]

        print("Registration Complete.")

    def __recognition(self):
        if self.frame is None:
            return

        self.THREAD["recog"] = 1  # LOCK

        if self.frame_num % self.CONFIG["process_frame"] == 0:
            image = self.frame.copy()
            image = cv2.resize(
                image,
                (0, 0),
                fx=self.CONFIG["recog_size"]["fx"],
                fy=self.CONFIG["recog_size"]["fy"],
            )

            face_locations = face.face_locations(image)
            face_encodings = face.face_encodings(image, face_locations)

            if len(face_locations) == 0:
                self.recognized_face = []

                if "no_face" not in self.CHECK:
                    self.NO_FACE = 1
                else:
                    self.NO_FACE += 1

                if self.NO_FACE >= self.CONFIG["check_no_face"]:
                    # Speak
                    if self.THREAD["voice"] == 0:
                        voice_thread = self.__set_thread(
                            target=self.__voice,
                            kwargs_dict={"name_list": ["no_face_find"]},
                        )
                        voice_thread.start()

                    self.NO_FACE = 0

                self.THREAD["recog"] = 0
                return

            face_info_list = []
            recogized_face_name = [n for n, l in self.recognized_face]
            for idx, encoding in enumerate(face_encodings):
                name = "unknown"
                if len(self.encoding) != 0:
                    distance = face.face_distance(
                        list(self.encoding.values()), encoding
                    )
                    match_idx = np.argmin(distance)

                    if distance[match_idx] < self.CONFIG["distance_thres"]:
                        name = list(self.encoding.keys())[match_idx]

                face_info = (name, face_locations[idx])
                face_info_list.append(face_info)

            self.recognized_face = face_info_list

            # Speak
            if self.THREAD["voice"] == 0:
                voice_thread = self.__set_thread(
                    target=self.__voice, kwargs_dict={"name_list": self.recognized_face}
                )
                voice_thread.start()

        self.THREAD["recog"] = 0  # Release

    def __loading(self):
        print("Loading . . . ")

        # Element : (image_path, names)
        database = [
            (str(d), str(d).split("/")[-1].split(".")[0])
            for d in Path(self.CONFIG["database_dir"]).glob("*")
        ]

        encoding = OrderedDict()
        for img_path, name in database:
            image = face.load_image_file(img_path)
            face_locations = face.face_locations(image)
            encoding[name] = face.face_encodings(image, face_locations)[0]

        print(f"{len(database)} images loaded.")
        print("Start Face Recognition.")

    def __voice(self, name_list):
        self.THREAD["voice"] = 1

        for name in name_list:
            voice_file = os.path.join(self.CONFIG["sound_dir"], name + ".mp3")
            os.system("mpg123 " + voice_file)
            time.sleep(self.CONFIG["voice_interval"])

        time.sleep(self.CONFIG["voice_delay"])

        self.THREAD["voice"] = 0

    def __set_thread(self, target, kwargs_dict={}):
        return threading.Thread(target=target, kwargs=kwargs_dict)
