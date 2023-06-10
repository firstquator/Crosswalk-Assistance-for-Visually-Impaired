import cv2
import numpy as np
import math
import time

# from PIL import Image, ImageFont, ImageDraw
from sklearn import linear_model
from YOLO.YOLO import YOLO

# TODO #
# 1. debug mode
# 2. 이미지에 이쁘게 글씨 넣는 법 검색


class HELP_CROSSWALK:
    CONFIG = {
        "YOLO": {
            "weight": "crosswalk_n.onnx",
            "conf_thres": 0.2,
            "iou_thres": 0.3,
        },
        "FIND_ZC": {
            "min_region_area_factor": 0.05,
            "max_region_area_factor": 0.1,
            "iou_thres": 0.2,
            "interval": 50,
            "timer": -1,
            "pause_time": 3,
            "error_cnt": 0,
            "debug_color_dict": {"font_color": (0, 0, 0)},
        },
        "LOCATION_PED": {
            "min_region_area_factor": 0.05,
            "max_region_area_factor": 0.2,
            "hsv_white": [[170, 170, 170], [255, 255, 255]],
            "erode_rate": 15,
            "contour_area_thres": 0.010,
            "width_thres": 50,
            "radius_thres": 250,
            "safe_interval": 85,
            "safe_angle": 30,
            "min_points": 3,
            "timer": -1,
            "pause_time": 3,
            "debug_color_dict": {
                "font_color": (0, 0, 0),
                "connect_lines": (212, 250, 252),
                "left_points": (184, 231, 225),
                "right_points": (189, 205, 214),
                "center_points": (241, 247, 181),
                "frame_center": (250, 220, 100),
                "x_interpolation": (250, 250, 175),
                "safety_zones": (138, 138, 253),
            },
        },
        "DETECT_TL": {},
    }

    CLASS = {"ZC": 0, "RED": 1, "GREEN": 2}
    MODE = {
        "FIND_ZC": 0,
        "LOCATION_PED": 1,
        "DETECT_TRAFFIC_LIGHT": 2,
        "HELP_CROSSWALK": 3,
    }
    HISTORY = {"FIND_ZC": [], "LOCATION_PED": [], "DETECT_TRAFFIC_LIGHT": []}

    def __init__(self, debug: bool = False):
        self.detector = self.__prepare_yolo()
        self.cur_mode = self.MODE["FIND_ZC"]
        self.debug = debug

    def __call__(self, frame: np.ndarray, mode: int = 3):
        if mode == self.MODE["FIND_ZC"]:
            return self.find_zebra_crossing(frame)
        elif mode == self.MODE["LOCATION_PED"]:
            return self.location_ped(frame)
        elif mode == self.MODE["DETECT_TRAFFIC_LIGHT"]:
            return self.detect_traffic_light(frame)

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
        # r_cy, r_cx = r_H // 2, r_W // 2

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

        # 2. If the bbox is at center, represent to change mode
        duration = 0
        if (
            ud_idx == 1
            and lr_idx == 1
            and r_area > W * H * self.CONFIG["FIND_ZC"]["max_region_area_factor"]
        ):
            alarm = "MOVE : STOP"
            if self.CONFIG["FIND_ZC"]["timer"] == -1:
                self.CONFIG["FIND_ZC"]["timer"] = time.time()
            else:
                duration = round(time.time() - self.CONFIG["FIND_ZC"]["timer"], 3)
                if duration > self.CONFIG["FIND_ZC"]["pause_time"]:
                    self.cur_mode = self.MODE["LOCATION_PED"]
                    self.HISTORY["FIND_ZC"].clear()
        else:
            self.CONFIG["FIND_ZC"]["timer"] = -1

        # [ DEBUG ]
        if self.debug:
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

        ## Compute middle points
        ransacMiddle = (ransacLeft + ransacRight) // 2

        if len(ransacLeft) < self.CONFIG["LOCATION_PED"]["min_points"]:
            return frame

        # 7. Calculate the intersection point of the bounding lines

        ## Unit vector + A point on each line
        vx_M, vy_M, x0_M, y0_M = cv2.fitLine(ransacMiddle, cv2.DIST_L2, 0, 0.01, 0.01)

        ## Get y = mx + b
        m_M, b_M = self.__lineCalc(vx_M, vy_M, x0_M, y0_M)

        ## Calculate x-interpolation
        x_interpolation = (H - b_M) // m_M
        x_interpolation = int(x_interpolation)

        # 8. Calculate the direction vector
        Cx = (W - 1) // 2  # Center of the screen

        # 9. Calculate the angle
        left, right = ransacLeft[0], ransacRight[0]
        angle = self.__angleCalc(left, right)

        # 10. Control the user directions to get to the correct location.
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
            if angle > 0:
                turn_idx = 0
            elif angle < 0:
                turn_idx = 2
        else:
            turn_idx = 1

        ## Control User (Speaker)
        duration = 0
        if move_idx == 1 and turn_idx == 1:
            if self.CONFIG["LOCATION_PED"]["timer"] == -1:
                self.CONFIG["LOCATION_PED"]["timer"] = time.time()
            else:
                duration = round(time.time() - self.CONFIG["LOCATION_PED"]["timer"], 3)
                if duration > self.CONFIG["LOCATION_PED"]["pause_time"]:
                    self.cur_mode = self.MODE["DETECT_TRAFFIC_LIGHT"]
                    self.HISTORY["LOCATION_PED"].clear()
        else:
            self.CONFIG["LOCATION_PED"]["timer"] = -1

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
                    2,
                )  ### Circles -> middle line

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
        pass

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
        x_left = max(x1_box1, x1_box2)
        y_top = max(y1_box1, y1_box2)
        x_right = min(x2_box1, x2_box2)
        y_bottom = min(y2_box1, y2_box2)

        # Calculate the width and height of nested boxes
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # Compute IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

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

        return np.degrees(np.arctan2(dy, dx))
