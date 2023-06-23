import os
import cv2
import time
import threading
import numpy as np

# from PIL import Image, ImageFont, ImageDraw
from sklearn import linear_model
from YOLO.YOLO import YOLO


class HELP_CROSSWALK:
    CONFIG = {
        "YOLO": {
            "weight": "crosswalk_n.onnx",
            "conf_thres": 0.2,
            "iou_thres": 0.3,
        },
        "VOICE": {
            "delay": 1,
        },
        "FIND_ZC": {
            "min_region_area_factor": 0.05,
            "max_region_area_factor": 0.1,
            "iou_thres": 0.1,
            "interval": 50,
            "timer": -1,
            "pause_time": 3,
            "error_cnt": 0,
            "debug_color_dict": {
                "font_color": (255, 0, 0),
                "not_satisfied_area": (75, 77, 235),
                "satisfied_area": (76, 176, 106),
            },
        },
        "LOCATION_PED": {
            "min_region_area_factor": 0.05,
            "max_region_area_factor": 0.2,
            "hsv_white": [[170, 170, 170], [255, 255, 255]],
            "erode_rate": 21,
            "contour_area_thres": 0.010,
            "width_thres": 50,
            "radius_thres": 250,
            "safe_interval": 85,
            "safe_angle": 30,
            "min_points": 3,
            "timer": -1,
            "pause_time": 3,
            "debug_color_dict": {
                "font_color": (255, 0, 0),
                "connect_lines": (34, 190, 242),
                "left_points": (39, 151, 242),
                "right_points": (39, 151, 242),
                "center_points": (61, 76, 242),
                "frame_center": (100, 220, 250),
                "x_interpolation": (96, 0, 255),
                "safety_zones": (255, 121, 0),
            },
        },
        "DETECT_TL": {
            "cumulative_light": 5,
            "class_rate": 0.8,
            "timer": -1,
            "delay": 20,
            "red_switch": 0,
            "debug_color_dict": {
                "font_color": (255, 0, 0),
            },
        },
    }

    THREAD = {
        "SOFT": 0,
        "HARD": 0,
    }  # Shared variables in threads (0 : Accessible, 1 : Inaccessible)
    CLASS = {"ZC": 0, "RED": 1, "GREEN": 2}
    MODE = {
        "FIND_ZC": 0,
        "LOCATION_PED": 1,
        "DETECT_TRAFFIC_LIGHT": 2,
        "HELP_CROSSWALK": 3,
    }
    HISTORY = {"FIND_ZC": [], "LOCATION_PED": [], "DETECT_TL": []}
    VOICE = {
        # Only Move
        "STOP": "./sound/crosswalk/stop.mp3",
        "STRAIGHT": "./sound/crosswalk/straight.mp3",
        "BACK": "./sound/crosswalk/back.mp3",
        "LEFT": "./sound/crosswalk/left.mp3",
        "RIGHT": "./sound/crosswalk/right.mp3",
        "UP": "./sound/crosswalk/up.mp3",
        "DOWN": "./sound/crosswalk/down.mp3",
        "UPLEFT": "./sound/crosswalk/upleft.mp3",
        "UPRIGHT": "./sound/crosswalk/upright.mp3",
        "DOWNLEFT": "./sound/crosswalk/downleft.mp3",
        "DOWNRIGHT": "./sound/crosswalk/downright.mp3",
        # Move, Turn
        "LEFTLEFT": "./sound/crosswalk/leftleft.mp3",
        "LEFTRIGHT": "./sound/crosswalk/leftright.mp3",
        "RIGHTLEFT": "./sound/crosswalk/rightleft.mp3",
        "RIGHTRIGHT": "./sound/crosswalk/rightright.mp3",
        # Traffic light
        "RED": "./sound/crosswalk/red.mp3",
        "GREEN": "./sound/crosswalk/green.mp3",
        "NONE": "./sound/crosswalk/none.mp3",
    }

    def __init__(self, debug: bool = False, voice: bool = True):
        self.detector = self.__prepare_yolo()
        self.cur_mode = self.MODE["FIND_ZC"]
        self.debug = debug
        self.voice = voice

    def __call__(self, frame: np.ndarray, mode: int = 3):
        if mode == self.MODE["FIND_ZC"]:
            return self.find_zebra_crossing(frame)
        elif mode == self.MODE["LOCATION_PED"]:
            return self.location_ped(frame)
        elif mode == self.MODE["DETECT_TRAFFIC_LIGHT"]:
            return self.detect_traffic_light(frame)
        elif mode == self.MODE["HELP_CROSSWALK"]:
            return self.help_crosswalk(frame)

    ######################## Setter & Getter ########################
    def set_onnx(self, onnx_path: str):
        self.CONFIG["YOLO"]["weight"] = onnx_path

    def set_mode(self, mode: int):
        self.cur_mode = mode

    def get_mode(self):
        return self.cur_mode

    ########################## Main Blocks ##########################
    def help_crosswalk(self, frame: np.ndarray):
        if self.get_mode() == self.MODE["FIND_ZC"]:
            return self.find_zebra_crossing(frame)
        elif self.get_mode() == self.MODE["LOCATION_PED"]:
            return self.location_ped(frame)
        elif self.get_mode() == self.MODE["DETECT_TRAFFIC_LIGHT"]:
            return self.detect_traffic_light(frame)

    def find_zebra_crossing(self, frame: np.ndarray):
        # Obtain the required parameters.
        bbox, score, class_id = self.__get_detect_info(frame, class_id=self.CLASS["ZC"])
        if len(bbox) == 0:
            return frame

        # Get information about detected Zebra-Crossing region
        x1, y1, x2, y2 = bbox.astype(np.uint32)
        crop_region = frame[y1:y2, x1:x2, :]

        # Check for a sharp rate of change from the previous frame
        if len(self.HISTORY["FIND_ZC"]) > 0:
            if not self.__compare_iou(
                bbox.astype(np.uint32), thresh=self.CONFIG["FIND_ZC"]["iou_thres"]
            ):
                self.HISTORY["FIND_ZC"].clear()

                if self.debug:
                    print("📝 [FIND_ZC ERROR] Changes screen are too rapid.")
                return frame

            else:
                self.HISTORY["FIND_ZC"].append(bbox.astype(np.uint32))
        else:
            self.HISTORY["FIND_ZC"].append(bbox.astype(np.uint32))

        # Get the size of the frame and the size of the cropped frame
        H, W = frame.shape[:2]
        r_H, r_W = crop_region.shape[:2]

        f_cy, f_cx = H // 2, W // 2

        r_area = r_H * r_W
        if r_area < H * W * self.CONFIG["FIND_ZC"]["min_region_area_factor"]:
            return frame

        # 1. Find the Center orientation
        UD = ("UP", "STRAIGHT", "DOWN")
        LR = ("LEFT", "STRAIGHT", "RIGHT")

        interval = self.CONFIG["FIND_ZC"]["interval"]

        ## UP and DOWN
        if y2 < f_cy - interval:
            ud_idx = 0
        elif y1 > f_cy + interval:
            ud_idx = 2
        else:
            ud_idx = 1
        ## LEFT and RIGHT
        if x2 < f_cx - interval:
            lr_idx = 0
        elif x1 > f_cx + interval:
            lr_idx = 2
        else:
            lr_idx = 1

        ## To move it to the center, give it orientation information
        alarm = f"MOVE : {UD[ud_idx]} {LR[lr_idx]}"

        ud = UD[ud_idx] if ud_idx != 1 else ""
        lr = LR[lr_idx] if lr_idx != 1 else ""

        direction = ud + lr

        # 2. If the bbox is at center, represent to change mode
        duration = 0

        ## If the bbox is centered and reaches a certain distance, represent to change mode
        color = None
        if (
            ud_idx == 1
            and lr_idx == 1
            and r_area >= W * H * self.CONFIG["FIND_ZC"]["max_region_area_factor"]
        ):
            color = self.CONFIG["FIND_ZC"]["debug_color_dict"]["satisfied_area"]
            alarm = "MOVE : STOP"
            if self.voice and self.THREAD["HARD"] == 0:
                voice_thread = self.__set_voice_thread("STOP", strength="HARD")
                voice_thread.start()

            if self.CONFIG["FIND_ZC"]["timer"] == -1:
                self.CONFIG["FIND_ZC"]["timer"] = time.time()
            else:
                duration = round(time.time() - self.CONFIG["FIND_ZC"]["timer"], 3)
                if duration > self.CONFIG["FIND_ZC"]["pause_time"]:
                    self.cur_mode = self.MODE["LOCATION_PED"]
                    self.HISTORY["FIND_ZC"].clear()

        ## If the bbox is centered and has not reached a certain distance, tell it to go straight.
        elif (
            ud_idx == 1
            and lr_idx == 1
            and r_area < W * H * self.CONFIG["FIND_ZC"]["max_region_area_factor"]
        ):
            color = self.CONFIG["FIND_ZC"]["debug_color_dict"]["not_satisfied_area"]
            if self.voice and self.THREAD["SOFT"] == 0:
                voice_thread = self.__set_voice_thread("STRAIGHT")
                voice_thread.start()
        else:
            color = self.CONFIG["FIND_ZC"]["debug_color_dict"]["not_satisfied_area"]
            self.CONFIG["FIND_ZC"]["timer"] = -1
            if self.voice and self.THREAD["SOFT"] == 0:
                voice_thread = self.__set_voice_thread(direction)
                voice_thread.start()

        # [ DEBUG ]
        if self.debug:
            frame = self.detector.draw_detections(
                frame, ([bbox], [score], [class_id]), color
            )  ## Draw bbox

            cv2.putText(
                frame,
                f"Duration : {duration}",
                (10, 70),  # (x, y)
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.CONFIG["FIND_ZC"]["debug_color_dict"]["font_color"],
                2,
                cv2.LINE_AA,
            )  ## Show elapsed time when ZC is centered
            cv2.putText(
                frame,
                alarm,
                (10, 100),  # (x, y)
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.CONFIG["FIND_ZC"]["debug_color_dict"]["font_color"],
                2,
                cv2.LINE_AA,
            )  ## Shows the direction to move to center the Zebra-Crossing

        return frame

    def location_ped(self, frame: np.ndarray):
        # Obtain the required parameters.
        bbox, score, class_id = self.__get_detect_info(frame, class_id=self.CLASS["ZC"])
        if len(bbox) == 0:
            return frame

        # Get information about detected Zebra-Crossing region
        x1, y1, x2, y2 = bbox.astype(np.uint32)
        crop_region = frame[y1:y2, x1:x2, :]

        # Get the size of the frame and the size of the cropped frame
        H, W = frame.shape[:2]
        r_H, r_W = crop_region.shape[:2]

        r_area = r_H * r_W
        if r_area < H * W * self.CONFIG["LOCATION_PED"]["min_region_area_factor"]:
            return frame

        # 1. Filter the white Color
        lower = np.array(self.CONFIG["LOCATION_PED"]["hsv_white"][0])
        upper = np.array(self.CONFIG["LOCATION_PED"]["hsv_white"][1])
        mask = cv2.inRange(crop_region, lower, upper)

        # 2. Erode the image
        erode_size = int(r_H / self.CONFIG["LOCATION_PED"]["erode_rate"])
        erode_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, 1))
        erode = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, erode_structure)

        # 3. Find contours & Draw the lines on the white stripes

        ## Collect stripe's bounding box coordinates
        bxbyLeftArray, bxbyRightArray = [], []

        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if (
                cv2.contourArea(contour)
                < r_area * self.CONFIG["LOCATION_PED"]["contour_area_thres"]
            ):
                continue

            if self.debug:
                cv2.drawContours(erode, [contour], 0, (255, 0, 0), 3)

            rect = cv2.minAreaRect(contour)
            left, right = self.__boxPoints(rect)
            left += np.array([x1, y1])
            right += np.array([x1, y1])

            if (
                self.__l2_distance(left, right)
                > self.CONFIG["LOCATION_PED"]["width_thres"]
            ):
                bxbyLeftArray.append(left)  # x, y for the left line
                bxbyRightArray.append(right)  # x, y for the right line

        # 4. Calculate median average for each line
        bxbyLeftArray = np.asarray(bxbyLeftArray)
        bxbyRightArray = np.asarray(bxbyRightArray)

        medianL = np.median(bxbyLeftArray, axis=0)
        medianR = np.median(bxbyRightArray, axis=0)

        # 5. Check the points bounded within the median circle
        boundedLeft, boundedRight = [], []
        for left, right in zip(bxbyLeftArray, bxbyRightArray):
            checkL, checkR = False, False

            if (
                self.__l2_distance(medianL, left)
                < self.CONFIG["LOCATION_PED"]["radius_thres"]
            ):
                checkL = True
            if (
                self.__l2_distance(medianR, right)
                < self.CONFIG["LOCATION_PED"]["radius_thres"]
            ):
                checkR = True

            if checkL and checkR:
                boundedLeft.append(left)
                boundedRight.append(right)

        boundedLeft = np.array(boundedLeft)
        boundedRight = np.array(boundedRight)
        if len(boundedLeft) < self.CONFIG["LOCATION_PED"]["min_points"]:
            return frame

        # 6. RANSAC Algorithm

        ## Select the points enclosed within the circle (from the last part)
        bxLeft, byLeft = boundedLeft[:, 0], boundedLeft[:, 1]
        bxRight, byRight = boundedRight[:, 0], boundedRight[:, 1]
        bxLeftT, bxRightT = np.array([bxLeft]).T, np.array([bxRight]).T

        if len(bxLeftT) < self.CONFIG["LOCATION_PED"]["min_points"]:
            return frame

        ## Run ransac for LEFT, RIGHT
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

        ransacX = model_ransac.fit(bxLeftT, byLeft)
        inlier_maskL = model_ransac.inlier_mask_  # Left ransac mask

        ransacY = model_ransac.fit(bxRightT, byRight)
        inlier_maskR = model_ransac.inlier_mask_  # Right ransc mask

        inlier_mask_or = inlier_maskL + inlier_maskR
        ransacLeft, ransacRight = (
            boundedLeft[inlier_mask_or],
            boundedRight[inlier_mask_or],
        )

        # 7. Pruning ransac points
        positive, negative, length = [], [], []  # /, \
        for left, right in zip(ransacLeft, ransacRight):
            angle = self.__angleCalc(left, right)
            length = self.__l2_distance(left, right)
            # To match the shape of left and right
            length = np.array([length, length]).astype(np.int64)
            line_info = [left, right, length]

            if angle < 0:
                negative.append(line_info)
            elif angle > 0:
                positive.append(line_info)

        main_points = positive if len(positive) >= len(negative) else negative
        main_points = np.array(main_points)

        if len(main_points) < self.CONFIG["LOCATION_PED"]["min_points"]:
            return frame

        ## Pruning with angles

        ## Pruning with length
        lengths = main_points[:, 2]
        max_length = np.max(lengths)
        length_thres = max_length // 2
        length_mask = lengths > length_thres

        ransacLeft, ransacRight = (
            main_points[:, 0][length_mask[:, 0]],
            main_points[:, 1][length_mask[:, 0]],
        )

        if len(ransacLeft) < self.CONFIG["LOCATION_PED"]["min_points"]:
            return frame

        # 8. Calculate the intersection point of the bounding lines

        ## Compute middle points
        ransacMiddle = (ransacLeft + ransacRight) // 2

        ## Unit vector + A point on each line
        vx_M, vy_M, x0_M, y0_M = cv2.fitLine(ransacMiddle, cv2.DIST_L2, 0, 0.01, 0.01)

        ## Get y = mx + b
        m_M, b_M = self.__lineCalc(vx_M, vy_M, x0_M, y0_M)

        ## Calculate x-interpolation
        x_interpolation = (H - b_M) // m_M
        x_interpolation = int(x_interpolation)

        # 9. Calculate the direction vector
        Cx = (W - 1) // 2  # Center of the screen

        # 10. Calculate the angle
        left, right = ransacLeft[0], ransacRight[0]
        angle = self.__angleCalc(left, right)

        # 11. Control the user directions to get to the correct location.
        history = self.HISTORY["LOCATION_PED"]
        move = ("LEFT", "STOP", "RIGHT")
        turn = ("LEFT", "STOP", "RIGHT")

        ## Move
        if x_interpolation < Cx - self.CONFIG["LOCATION_PED"]["safe_interval"]:
            move_idx = 0
        elif x_interpolation > Cx + self.CONFIG["LOCATION_PED"]["safe_interval"]:
            move_idx = 2
        else:
            move_idx = 1

        ## Turn
        if abs(angle) > self.CONFIG["LOCATION_PED"]["safe_angle"]:
            if angle < 0:
                turn_idx = 0
            elif angle > 0:
                turn_idx = 2
        else:
            turn_idx = 1

        ## Control User (Voice)
        duration = 0
        mv = move[move_idx] if move_idx != 1 else ""
        tr = turn[turn_idx] if turn_idx != 1 else ""
        motion = mv + tr

        if move_idx == 1 and turn_idx == 1:
            ### Tell it to stop (Voice)
            if self.voice and self.THREAD["HARD"] == 0:
                voice_thread = self.__set_voice_thread("STOP", strength="HARD")
                voice_thread.start()

            if self.CONFIG["LOCATION_PED"]["timer"] == -1:
                self.CONFIG["LOCATION_PED"]["timer"] = time.time()

            else:
                duration = round(time.time() - self.CONFIG["LOCATION_PED"]["timer"], 3)
                if duration > self.CONFIG["LOCATION_PED"]["pause_time"]:
                    self.cur_mode = self.MODE["DETECT_TRAFFIC_LIGHT"]
                    self.HISTORY["LOCATION_PED"].clear()
        else:
            self.CONFIG["LOCATION_PED"]["timer"] = -1
            if self.voice and self.THREAD["SOFT"] == 0:
                voice_thread = self.__set_voice_thread(motion)
                voice_thread.start()

        # [ DEBUG ]
        if self.debug:
            cv2.imshow("LOCATION_PED : Mask", erode)

            ## Duration
            cv2.putText(
                frame,
                f"Duration : {duration}",
                (10, 30),  # (x, y)
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["font_color"],
                2,
                cv2.LINE_AA,
            )
            ## Show control
            control = f"Move : {move[move_idx]} / Turn : {turn[turn_idx]}"
            cv2.putText(
                frame,
                control,
                (10, 50),  # (x, y)
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["font_color"],
                2,
                cv2.LINE_AA,
            )

            ## Show the middle part of the left and right sides of a zebra-crossing
            for left, right, mid in zip(ransacLeft, ransacRight, ransacMiddle):
                cv2.line(
                    frame,
                    (left[0], left[1]),
                    (right[0], right[1]),
                    self.CONFIG["LOCATION_PED"]["debug_color_dict"]["connect_lines"],
                    2,
                )  ### Line connecting left and right points
                cv2.circle(
                    frame,
                    (left[0], left[1]),
                    5,
                    self.CONFIG["LOCATION_PED"]["debug_color_dict"]["left_points"],
                    2,
                )  ### Circles -> left line
                cv2.circle(
                    frame,
                    (right[0], right[1]),
                    5,
                    self.CONFIG["LOCATION_PED"]["debug_color_dict"]["right_points"],
                    2,
                )  ### Circles -> right line
                cv2.circle(
                    frame,
                    (mid[0], mid[1]),
                    5,
                    self.CONFIG["LOCATION_PED"]["debug_color_dict"]["center_points"],
                    1,
                )  ### Circles -> middle line

                cv2.arrowedLine(
                    frame,
                    (x_interpolation, H - 1),
                    (ransacMiddle[-1]),
                    self.CONFIG["LOCATION_PED"]["debug_color_dict"]["x_interpolation"],
                    10,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

            ## Show center point, x interpolation point and safety zones
            cv2.circle(
                frame,
                (Cx, H - 1),
                7,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["frame_center"],
                10,
            )  ### Center points
            cv2.circle(
                frame,
                (x_interpolation, H - 1),
                3,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["x_interpolation"],
                10,
            )  ### show x interpolation point
            cv2.circle(
                frame,
                (Cx - self.CONFIG["LOCATION_PED"]["safe_interval"], H - 1),
                5,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["safety_zones"],
                10,
            )  ### Left safety point
            cv2.circle(
                frame,
                (Cx + self.CONFIG["LOCATION_PED"]["safe_interval"], H - 1),
                5,
                self.CONFIG["LOCATION_PED"]["debug_color_dict"]["safety_zones"],
                10,
            )  ### Right safety point

        return frame

    def detect_traffic_light(self, frame: np.ndarray):
        # Obtain the required parameters.
        bbox_red = self.__get_detect_info(frame, class_id=self.CLASS["RED"])
        bbox_green = self.__get_detect_info(frame, class_id=self.CLASS["GREEN"])

        light_class = ("RED", "GREEN", "NONE")

        if bbox_red:
            light_idx = 0
        elif bbox_green:
            light_idx = 1
        else:
            light_idx = 2

        traffic_light = None
        cumulate = self.HISTORY["DETECT_TL"]

        if len(cumulate) > self.CONFIG["DETECT_TL"]["cumulative_light"]:
            most_class = max(set(cumulate), key=cumulate.count)
            num_of_class = cumulate.count(most_class)
            if (
                num_of_class
                > self.CONFIG["DETECT_TL"]["cumulative_light"]
                * self.CONFIG["DETECT_TL"]["class_rate"]
            ):
                traffic_light = light_class[most_class]
                timer = False

                if traffic_light == "NONE":
                    timer = True
                elif traffic_light == "RED":
                    self.CONFIG["DETECT_TL"]["red_switch"] = 1
                    timer = False
                elif traffic_light == "GREEN":
                    if self.CONFIG["DETECT_TL"]["red_switch"] == 1:
                        timer = True
                        self.CONFIG["DETECT_TL"]["red_switch"] = 0
                    else:
                        timer = False

                if timer:
                    if self.CONFIG["DETECT_TL"]["timer"] == -1:
                        self.CONFIG["DETECT_TL"]["timer"] = time.time()
                    else:
                        if (
                            time.time() - self.CONFIG["DETECT_TL"]["timer"]
                            > self.CONFIG["DETECT_TL"]["delay"]
                        ):
                            self.cur_mode = self.MODE["FIND_ZC"]
                            self.CONFIG["DETECT_TL"]["timer"] = -1

                # Voice
                if self.voice and self.THREAD["HARD"] == 0:
                    voice_thread = self.__set_voice_thread(
                        traffic_light, strength="HARD"
                    )
                    voice_thread.start()

                cumulate.clear()
                traffic_light = None
            else:
                cumulate.clear()
                return frame

        else:
            cumulate.append(light_idx)

        # [ DEBUG ]
        if self.debug:
            if traffic_light == None:
                light_name = light_class[light_idx]
            else:
                light_name = traffic_light
            cv2.putText(
                frame,
                light_name,
                (10, 50),  # (x, y)
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.CONFIG["DETECT_TL"]["debug_color_dict"]["font_color"],
                2,
                cv2.LINE_AA,
            )

        return frame

    ########################## Private Functions ##########################
    def __prepare_yolo(self):
        """
        Set the object detector, YOLO.

        """
        yolo_onnx_path = f"./YOLO/models/{self.CONFIG['YOLO']['weight']}"
        yolo_detector = YOLO(
            yolo_onnx_path,
            conf_thres=self.CONFIG["YOLO"]["conf_thres"],
            iou_thres=self.CONFIG["YOLO"]["iou_thres"],
        )

        return yolo_detector

    def __voice(self, select_voice: str, strength):
        self.THREAD[strength] = 1
        if strength == "HARD":
            self.THREAD["SOFT"] = 1

        voice_file = self.VOICE[select_voice.upper()]  # mp3 or mid file

        os.system("mpg123 " + voice_file)

        if strength == "HARD":
            time.sleep(self.CONFIG["FIND_ZC"]["pause_time"])
        elif strength == "SOFT":
            time.sleep(self.CONFIG["VOICE"]["delay"])

        self.THREAD[strength] = 0
        if strength == "HARD":
            self.THREAD["SOFT"] = 0

    def __set_voice_thread(self, select_voice: str, strength: str = "SOFT"):
        return threading.Thread(
            target=self.__voice,
            kwargs={"select_voice": select_voice, "strength": strength},
        )

    def __get_detect_info(self, frame: np.ndarray, class_id):
        # 1. Find a zebra-crossing using YOLO.
        boxes, scores, class_ids = self.detector(frame)
        if len(boxes) == 0:
            return [[], [], []]

        # 2. Filter only the BBox information you need.
        stripes, light_red, light_green = self.__filter_info(
            np.c_[boxes, scores, class_ids]
        )

        if class_id == self.CLASS["ZC"]:
            return stripes
        elif class_id == self.CLASS["RED"]:
            return light_red
        elif class_id == self.CLASS["GREEN"]:
            return light_green

    def __filter_info(self, detect_info_npy: np.ndarray):
        # Filtering : if Each zebra crossing(Stripe), traffic light have two pieces of information, reduce it to one
        class_ids = detect_info_npy[:, 5]
        stripes = detect_info_npy[class_ids == 0]
        light_red = detect_info_npy[class_ids == 1]
        light_green = detect_info_npy[class_ids == 2]

        # Zebra-Crossing ( Stripes )
        if len(stripes) > 1:
            x1 = stripes[:, 0]
            max_x1_idx = x1.argmax(axis=0)
            stripes = np.expand_dims(stripes[max_x1_idx, :], axis=0)

        # RED Traffic Light
        if len(light_red) > 1:
            scores = light_red[:, 4]
            max_scores_idx = scores.argmax(axis=0)
            light_red = np.expand_dims(light_red[max_scores_idx, :], axis=0)

        # GREEN Traffic Light
        if len(light_green) > 1:
            scores = light_green[:, 4]
            max_scores_idx = scores.argmax(axis=0)
            light_green = np.expand_dims(light_green[max_scores_idx, :], axis=0)

        stripes = (
            [stripes[0, :4], stripes[0, 4], stripes[0, 5]]
            if len(stripes) != 0
            else stripes
        )
        light_red = (
            [light_red[0, :4], light_red[0, 4], light_red[0, 5]]
            if len(light_red) != 0
            else []
        )
        light_green = (
            [light_green[0, :4], light_green[0, 4], light_green[0, 5]]
            if len(light_green) != 0
            else []
        )

        if light_red and light_green:
            light_red, light_green = [], []

        return stripes, light_red, light_green

    def __compare_iou(self, cur: np.ndarray, thresh: float, objects: str = "FIND_ZC"):
        prev = self.HISTORY[objects][-1]
        iou = self.__calculate_iou(prev, cur)

        return True if iou > thresh else False

    def __calculate_iou(self, box1, box2):
        """
        Calculates the IoU between two boxes.

        [ Parameters ]
        box1 (list): Coordinates of the first box [x1, y1, x2, y2]
        box2 (list): Coordinates of the second box [x1, y1, x2, y2]

        [ Return ]
        float: IoU

        """

        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
        box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

        # Calculate coordinate overlap of boxes
        x_left = np.maximum(x1_box1, x1_box2)
        y_top = np.maximum(y1_box1, y1_box2)
        x_right = np.minimum(x2_box1, x2_box2)
        y_bottom = np.minimum(y2_box1, y2_box2)

        # Calculate the width and height of nested boxes
        intersection_area = np.maximum(0, x_right - x_left + 1) * np.maximum(
            0, y_bottom - y_top + 1
        )
        union_area = box1_area + box2_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def __boxPoints(self, rect):
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        base = box[0]
        coords = []

        for i, point in enumerate(box[1:]):
            coords.append([i + 1, self.__l2_distance(base, point)])

        coords = sorted(coords, key=lambda x: x[1])
        left = (box[0] + box[coords[0][0]]) // 2
        right = (box[coords[1][0]] + box[coords[2][0]]) // 2

        return left, right

    def __l2_distance(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def __lineCalc(self, vx, vy, x0, y0):
        # scale = 10

        # x1 = x0 + scale * vx
        # y1 = y0 + scale * vy

        if vx == 0:
            return 1, 0

        # m = (y1 - y0) / (x1 - x0)   # Slope
        # b = y1 - m * x1             # Y-intercept

        m = vy / vx
        b = y0 - m * x0

        return m, b

    def __angleCalc(self, pt1, pt2):
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]

        return np.degrees(np.arctan2(dy, dx)) * -1
