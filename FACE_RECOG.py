import os
import cv2
import time
import threading
import numpy as np
import face_recognition as face

from gtts import gTTS
from pathlib import Path
from collections import OrderedDict


class FACE_RECOG:
    CONFIG = {
        "database_dir": "./FACE_LIST/",
        "sound_dir": "./sound/face/",
        "capture_size": {"fx": 1.0, "fy": 1.0},
        "recog_size": {"fx": 0.25, "fy": 0.25},
        "distance_thres": 0.4,
        "process_frame": 8,
        "check_face": 5,
        "voice_interval": 1.2,
        "voice_delay": 3,
    }

    CHECK = {"location_list": []}
    THREAD = {"voice": 0, "register": 0}

    def __init__(self, debug=False):
        self.debug = debug
        self.encoding = None
        self.__loading()

    def __call__(self, frame, frame_num, key):
        self.face_recognition(self, frame, frame_num, key=key)

    def face_recognition(self, frame, frame_num, key=None):
        if key & 0xFF == 114:  # R
            if self.THREAD["register"] == 0:
                register = self.__set_thread(
                    target=self.__register, kwards_dict={"frame": frame}
                )
                register.start()

        self.__recognition(frame, frame_num)

    def set_config(self, param, kwards):
        self.CONFIG[param] = kwards

    ########################## Private Functions ##########################
    def __register(self, frame):
        self.THREAD["register"] = 1

        image = frame.copy()
        image_capture = cv2.resize(
            image,
            (0, 0),
            fx=self.CONFIG["capture_size"]["fx"],
            fy=self.CONFIG["capture_size"]["fy"],
        )

        # Check Error : No face or multiple faces
        face_location = face.face_locations(image_capture)
        if len(face_location) != 1:
            if self.debug:
                print(
                    f"I found {len(face_location)} face(s) in this photograph. so capture canceled."
                )

            return frame

        # Input name
        name = input("Please Enter a name : ")

        # Save images
        save_path = os.path.join(self.CONFIG["database_dir"], name + ".jpg")
        cv2.imwrite(save_path, image)

        # Save name voice
        save_path = os.path.join(self.CONFIG["sound_dir"])

        tts = gTTS(name, lang="en")
        tts.save(save_path)

        # Encoding New face
        self.encoding[name] = face.face_encodings(image)[0]

        self.THREAD["register"] = 0

    def __recognition(self, frame, frame_num):
        if frame_num % self.CONFIG["process_frame"] != 0:
            return frame

        image = frame.copy()
        image = cv2.resize(
            image,
            (0, 0),
            fx=self.CONFIG["recog_size"]["fx"],
            fy=self.CONFIG["recog_size"]["fy"],
        )

        face_locations = face.face_locations(image)
        face_encodings = face.face_encodings(image, face_locations)

        recognized_face = []
        for encoding in face_encodings:
            distance = face.face_distance(list(self.encoding.values()), encoding)
            match_idx = np.armin(distance)

            if distance[match_idx] < self.CONFIG["distance_thres"]:
                name = list(self.encoding.keys())[match_idx]
            else:
                name = "unknown"

            if name not in self.CHECK:
                self.CHECK[name] = 1
            else:
                self.CHECK[name] += 1
                if self.CHECK[name] >= self.CONFIG["check_face"]:
                    if self.debug:
                        print(f"Person Name : {name}")

                    recognized_face.append(name)
                    self.CHECK[name] = 0

        # Speak
        if len(face_locations) == 0:
            if self.THREAD["voice"] == 0:
                voice_thread = self.__set_thread(
                    target=self.__voice, kwards_dict={"name_list": ["no_face_find"]}
                )
                voice_thread.start()
        else:
            if self.THREAD["voice"] == 0:
                voice_thread = self.__set_thread(
                    target=self.__voice, kwards_dict={"name_list": recognized_face}
                )
                voice_thread.start()

            if self.debug:
                if len(recognized_face) == 0:
                    return frame

                for bbox, name in zip(face_locations, recognized_face):
                    # Rescaling
                    bbox = bbox * 1 / self.CONFIG["recog_size"]["fx"]

                    # Draw bbox
                    top, right, bottom, left = bbox
                    cv2.rectange(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectange(
                        frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom + 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        (255, 255, 255),
                        1,
                    )

        return frame

    def __loading(self):
        print(" Loading . . . ")

        # Element : (image_path, names)
        database = [
            (str(d), str(d).split("/")[-1].split(".")[0])
            for d in Path(self.CONFIG["database_dir"]).glob("*")
        ]

        encoding = OrderedDict()
        for img_path, name in database:
            image = face.load_image_file(img_path)
            encoding[name] = face.face_encodings(image)[0]

        print(f"{len(database)} images loaded.")

    def __voice(self, name_list):
        self.THREAD["voice"] = 1

        for name in name_list:
            voice_file = os.path.join(self.CONFIG["sound_dir"], name + ".mp3")
            os.system("mpg123 " + voice_file)
            time.sleep(self.CONFIG["voice_interval"])

        time.sleep(self.CONFIG["voice_delay"])

        self.THREAD["voice"] = 0

    def __set_thread(self, target, kwards_dict):
        return threading.Thread(target=target, kwards=kwards_dict)
