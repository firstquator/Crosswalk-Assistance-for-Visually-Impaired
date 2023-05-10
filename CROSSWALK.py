import cv2
import numpy as np
import math
import time
from sklearn import linear_model
from YOLO.YOLO import YOLO


class CROSSWALK:
    # HSV Range : Lower / Upper
    PARAM = {
        # Image Resize
        "img_size": (640, 640),
        "gaussian_filter": (5, 5),
        # Image Processing
        "hsv_white": [[170, 170, 170], [255, 255, 255]],
        "median_ksize": 5,
        "open_ksize": 3,
        # Detect white stripes
        "width_thresh": 50,
        "area_thresh": 1000,
        "radius_thresh": 250,
        # Find crosswalk parameters
        "error_cnt": 20,
        # Accumulate information
        "buffer_size": 6,
        # Safe parameter
        "safe_interval": 30,
        "safe_angle": 30,
        # YOLO Config
        "model_name": "crosswalk_n.onnx",
        "conf_thres": 0.2,
        "iou_thres": 0.3,
    }

    # Set YOLO Detector
    yolo_model_path = f'./YOLO/models/{PARAM["model_name"]}'
    yolo_detector = YOLO(
        yolo_model_path, conf_thres=PARAM["conf_thres"], iou_thres=PARAM["iou_thres"]
    )

    # Mode
    FIND_CROSSWALK = 0
    DETECT_CROSSWALK = 1
    DETECT_TRAFFIC_LIGHT = 2

    def __init__(self):
        # Help Crosswalk has 3 operation mode. Detalis => set_mode function
        # self.cur_mode = self.DETECT_CROSSWALK
        self.cur_mode = self.FIND_CROSSWALK

        # Continuous object detection information list
        self.idx = 0
        """ Find Crosswalk global values """
        self.find_crosswalk_box = []
        self.find_mode = 0
        self.timer = 0
        self.error_cnt = 0
        self.centerPrev = 0
        """ Help Crosswalk global values """
        self.data_buffer = []
        self.data_buffer_traffic = []
        self.DvAve = [0, 0]
        self.DvPrev = [0, 0]
        self.LightAve = 0
        self.LightPrev = 0
        self.guide = ""

    def __call__(self, frame):
        return self.detect(frame, display=True)

    def detect(self, frame: np.ndarray, display: bool = False):
        if frame.ndim < 3:
            assert "Frame must be 3 channels."

        # Resize frame
        frame = cv2.resize(frame, self.PARAM["img_size"])
        # Update object localizer & Detect objects
        boxes, scores, class_ids = self.yolo_detector(frame)
        if len(boxes) == 0:
            return 0, frame

        # Filtering info
        stripes, light_red, light_green = self.filter_info(
            np.c_[boxes, scores, class_ids]
        )
        box, score, class_id = stripes

        if self.get_mode() == self.FIND_CROSSWALK:
            self.find_crosswalk(frame, box, display=False)

        # Help Zebra-Crossing
        elif self.get_mode() == self.DETECT_CROSSWALK:
            if not stripes:
                return 0, frame

            frame, DirAngle = self.process_zebra_cross(frame, box, display=True)
            frame, guide_info = self.guide_to_crosswalk(frame, DirAngle, display=True)

            if guide_info:
                move = guide_info["move"].split(":")[-1].strip() == "STOP"
                turn = guide_info["turn"].split(":")[-1].strip() == "STOP"
            else:
                move, turn = False, False

            if move is True and turn is True:
                self.set_mode(self.DETECT_TRAFFIC_LIGHT)
                self.idx = 0

        # Detect Traffic Light
        elif self.get_mode() == self.DETECT_TRAFFIC_LIGHT:
            if light_red or light_green:
                status = self.check_traffic_light(frame, light_red or light_green)
            else:
                status = self.check_traffic_light(frame, None)

            if display is True:
                cv2.putText(
                    frame,
                    str(status),
                    (50, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        return 1, frame

    def find_crosswalk(self, frame: np.ndarray, bbox: np.ndarray, display=False):
        # Original frame shape
        h, w = frame.shape[:2]
        frame_center = w // 2

        # Check bbox
        if len(bbox) == 0:
            self.error_cnt += 1
            return

        # Crop ROI
        x1, y1, x2, y2 = bbox.astype("uint")

        roi = frame[y1:y2, x1:x2, :]
        roi_height, roi_width = roi.shape[:2]
        roi_center = (roi_height // 2, roi_width // 2)

        roi_area = roi_height * roi_width

        # Check ROI area
        if roi_area < h * w * 0.05:
            return

        # Location ) 0: Left, 1 : Straight, 2: Right

        turn = ["Left", "Center", "Right"]
        if x1 < frame_center and x2 < frame_center:
            turn_idx = 0
        elif x1 > frame_center and x2 > frame_center:
            turn_idx = 2
        else:
            turn_idx = 1

        # If certain errors occur, reset the mode to
        if self.error_cnt > self.PARAM["error_cnt"]:
            self.find_mode = 0

        # [ Mode : 0 ] Provide directions to crosswalks
        if self.find_mode == 0:
            duration = time.time() - self.timer
            if self.timer == 0:
                self.set_find_param(timer=time.time())
            else:
                # Collect orientation information up to 0.5 seconds before
                if duration < 0.5:
                    self.find_crosswalk_box.append(turn_idx)
                # Make sure this model detecting the right crosswalks
                elif duration > 0.5:
                    most_class = max(
                        set(self.find_crosswalk_box), key=self.find_crosswalk_box.count
                    )

                    if (
                        self.find_crosswalk_box.count(most_class)
                        / len(self.find_crosswalk_box)
                        >= 0.8
                    ):
                        self.set_find_param(mode=1)
                        self.find_crosswalk_box = [most_class]
                        self.centerPrev = roi_center
                        alarm = f"If you want to cross that crosswalk, center the crosswalk and hold still for 3 seconds.\n, then slowly turn to the {turn[most_class]}. We'll let you know when you're centered."
                        print(alarm)
                    else:
                        self.set_find_param(timer=0)
                        self.find_crosswalk_box = []

        # [ Mode : 1 ] Adjust the orientation of the crosswalk to be in the center position
        elif self.find_mode == 1:
            duration = time.time() - self.timer
            if self.timer == 0:
                self.set_find_param(timer=time.time(), mode=1)
            # Check if the previous ROI center coordinates are significantly different from the current center coordinates
            else:
                if self.l2_distance(self.centerPrev, roi_center) < 150:
                    self.find_crosswalk_box.append(turn_idx)
                    self.centerPrev = roi_center
                else:
                    self.error_cnt += 1

                if len(self.find_crosswalk_box) >= 5:
                    if (
                        max(
                            set(self.find_crosswalk_box[-5:]),
                            key=self.find_crosswalk_box.count,
                        )
                        == 1
                    ):
                        self.set_find_param(mode=2)
                        self.centerPrev = roi_center
                        self.find_crosswalk_box = []
                    elif duration > 3:
                        self.set_find_param(mode=0)
                        self.find_crosswalk_box = []

        # [ Mode : 2 ] Verify that you're trying to cross the street
        elif self.find_mode == 2:
            duration = time.time() - self.timer
            if roi_area < h * w * 0.1:
                self.set_find_param(timer=time.time(), mode=2)
                print("Go straight slowly.")

            else:
                if self.timer == 0:
                    self.set_find_param(timer=time.time(), mode=2)
                if duration < 3:
                    if self.l2_distance(self.centerPrev, roi_center) < 150:
                        self.find_crosswalk_box.append(turn_idx)
                    else:
                        self.error_cnt += 1
                else:
                    if (
                        max(
                            set(self.find_crosswalk_box[-5:-1]),
                            key=self.find_crosswalk_box.count,
                        )
                        == 1
                    ):
                        self.set_mode(self.DETECT_CROSSWALK)

    def process_zebra_cross(self, frame: np.ndarray, bbox: np.ndarray, display=False):
        # Original frame shape
        h, w = frame.shape[:2]

        # Crop ROI
        x1, y1, x2, y2 = bbox.astype("uint")

        roi = frame[y1:y2, x1:x2, :]
        roi_height, roi_width = roi.shape[:2]
        roi_area = roi_height * roi_width
        if roi_area < w * h * 0.05:
            return frame, [0, 0]

        # Collect stripe's bounding box coordinates
        bxbyLeftArray, bxbyRightArray = [], []

        # 1. Filter the white Color
        lower = np.array(self.PARAM["hsv_white"][0])
        upper = np.array(self.PARAM["hsv_white"][1])
        mask = cv2.inRange(roi, lower, upper)

        # 2. Erode the image
        erode_size = int(roi_height / 15)
        erode_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, 1))
        # erode = cv2.erode(mask, erode_structure, (-1, -1))
        erode = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, erode_structure)

        cv2.imshow("mask", erode)

        # 3. Find contours & Draw the lines on the white stripes
        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < roi_area * 0.010:
                continue

            rect = cv2.minAreaRect(contour)
            left, right = self.boxPoints(rect)
            left += np.array([x1, y1])
            right += np.array([x1, y1])

            # box = cv2.boxPoints(rect)
            # box = box + np.array([x1, y1])
            # box = np.int0(box)

            # frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            if self.l2_distance(left, right) > self.PARAM["width_thresh"]:
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

            if self.l2_distance(medianL, left) < self.PARAM["radius_thresh"]:
                checkL = True
            if self.l2_distance(medianR, right) < self.PARAM["radius_thresh"]:
                checkR = True

            if checkL and checkR:
                boundedLeft.append(left)
                boundedRight.append(right)

        boundedLeft = np.array(boundedLeft)
        boundedRight = np.array(boundedRight)
        if len(boundedLeft) < 3:
            return frame, [0, 0]

        # 6. RANSAC Algorithm

        # Select the points enclosed within the circle (from the last part)
        bxLeft, byLeft = boundedLeft[:, 0], boundedLeft[:, 1]
        bxRight, byRight = boundedRight[:, 0], boundedRight[:, 1]

        # Transpose x of the right and the left line
        bxLeftT, bxRightT = np.array([bxLeft]).T, np.array([bxRight]).T

        if len(bxLeftT) < 3:
            return frame, [0, 0]

        # Run ransac for LEFT, RIGHT
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

        ransacX = model_ransac.fit(bxLeftT, byLeft)
        inlier_maskL = model_ransac.inlier_mask_  # Left mask

        ransacY = model_ransac.fit(bxRightT, byRight)
        inlier_maskR = model_ransac.inlier_mask_  # Right mask

        inlier_mask_or = inlier_maskL + inlier_maskR
        ransacLeft, ransacRight = (
            boundedLeft[inlier_mask_or],
            boundedRight[inlier_mask_or],
        )
        if len(ransacLeft) < 3:
            return frame, [0, 0]

        # Remove outliers
        l2_box = [
            self.l2_distance(left, right)
            for left, right in zip(ransacLeft, ransacRight)
        ]
        outliers_idx = self.find_outliers(l2_box)

        if len(l2_box) - len(outliers_idx[outliers_idx == 0]) < 2:
            return frame, [0, 0]

        inlierLeft = ransacLeft[outliers_idx]
        inlierRight = ransacRight[outliers_idx]

        mean_l2_ang = []
        for left, right in zip(inlierLeft, inlierRight):
            mean_l2_ang.append(
                [self.l2_distance(left, right), self.angleCalc(left, right)]
            )

        m_l, m_a = np.array(mean_l2_ang).mean(axis=0)
        m_a = m_a * math.pi / 180

        new_left, new_right = [], []
        for idx, outlier, left, right in zip(
            np.arange(len(outliers_idx)), outliers_idx, ransacLeft, ransacRight
        ):
            if outlier == True:
                new_left.append(left)
                new_right.append(right)
            else:
                if inlier_maskL[idx] and inlier_maskR[idx]:
                    pass
                elif inlier_maskL[idx] == True:
                    new_left.append(left)

                    x, y = left[0] + m_l * np.cos(m_a), left[1] + m_l * np.sin(m_a)
                    new_right.append(np.array([x, y]))

                elif inlier_maskR[idx] == True:
                    new_right.append(right)

                    x, y = right[0] - m_l * np.cos(m_a), right[1] - m_l * np.sin(m_a)
                    new_left.append(np.array([x, y]))

        ransacLeft = np.array(new_left).astype(np.int32)
        ransacRight = np.array(new_right).astype(np.int32)

        # ransacLeft = ransacLeft[outliers_idx]
        # ransacRight = ransacRight[outliers_idx]

        # Compute middle points
        ransacMiddle = (ransacLeft + ransacRight) // 2

        # Parameter display = True ) Draw RANSAC selected circles
        if display == True:
            for left, right, mid in zip(ransacLeft, ransacRight, ransacMiddle):
                cv2.line(
                    frame, (left[0], left[1]), (right[0], right[1]), (212, 250, 252), 2
                )
                cv2.circle(
                    frame, (left[0], left[1]), 5, (184, 231, 225), 2
                )  # Circles -> left line
                cv2.circle(
                    frame, (right[0], right[1]), 5, (189, 205, 214), 2
                )  # Circles -> right line
                cv2.circle(
                    frame, (mid[0], mid[1]), 5, (241, 247, 181), 2
                )  # Circles -> middle line

        # 7. Calculate the intersection point of the bounding lines

        # Unit vector + A point on each line
        vx_M, vy_M, x0_M, y0_M = cv2.fitLine(ransacMiddle, cv2.DIST_L2, 0, 0.01, 0.01)

        # Get y = mx + b
        m_M, b_M = self.lineCalc(vx_M, vy_M, x0_M, y0_M)

        # Calculate x-interpolation
        x_interpolation = (h - b_M) // m_M
        x_interpolation = int(x_interpolation)

        # 8. Calculate the direction vector
        Cx = (w - 1) // 2  # Center of the screen
        dx = int(Cx - x_interpolation)  # Regular x axis coordinates

        if display == True:
            cv2.circle(frame, (Cx, h - 1), 7, (250, 220, 100), 10)
            cv2.circle(
                frame, (Cx - self.PARAM["safe_interval"], h - 1), 5, (138, 138, 253), 10
            )
            cv2.circle(
                frame, (Cx + self.PARAM["safe_interval"], h - 1), 5, (138, 138, 253), 10
            )

            cv2.circle(frame, (x_interpolation, h - 1), 3, (250, 250, 175), 10)
        # 9. Calculate the angle
        left, right = ransacLeft[0], ransacRight[0]
        angle = self.angleCalc(left, right)

        DirAngle = [dx, angle]  # Direction & Angle vector D = (dx, dy, angle)

        return frame, DirAngle

    def guide_to_crosswalk(
        self, frame: np.ndarray, DirAngle: np.ndarray, display: bool = False
    ):
        # Accumulate Direction & Angle information by PARAM['buffer_size']
        if self.idx < self.PARAM["buffer_size"]:
            self.data_buffer.append(DirAngle)
            self.idx += 1

        # Calculate Average information
        else:
            arr2np = np.array(self.data_buffer)
            self.DvAve = arr2np.mean(axis=0)

            self.idx = 0
            del self.data_buffer[:]

        safe_d, safe_a = self.PARAM["safe_interval"], self.PARAM["safe_angle"]
        DxAve, AngleAve = self.DvAve
        DxPrev, AnglePrev = self.DvPrev
        guide_info = {}

        if abs(DxAve) < self.PARAM["img_size"][0] // 2 - 20:
            # Check if the vanishing point and the next vanishing point aren't too far from each other
            if abs(DxPrev - DxAve) < 20 and abs(AnglePrev - AngleAve) < 20:
                # Moving guide
                if abs(DxAve) < safe_d:
                    move_dir = "MOVE : STOP"
                elif DxAve > safe_d:
                    move_dir = "MOVE : LEFT"
                elif DxAve < -safe_d:
                    move_dir = "MOVE : RIGHT"

                # Turn guide
                if AngleAve > -safe_a and AngleAve < safe_a:
                    turn_dir = "TURN : STOP"
                elif AngleAve > safe_a:
                    turn_dir = "TURN : LEFT"
                elif AngleAve < -safe_a:
                    turn_dir = "TURN : RIGHT"

                # Save guide information
                guide_info["move"] = move_dir
                guide_info["turn"] = turn_dir

                if display is True:
                    cv2.putText(
                        frame,
                        move_dir,
                        (50, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        turn_dir,
                        (50, 90),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            self.DvPrev = self.DvAve

        return frame, guide_info

    def check_traffic_light(self, frame: np.ndarray, light_info):
        NONE, RED, GREEN, COLLECTING = 0, 1, 2, 3
        if light_info == None:
            self.data_buffer_traffic.append(NONE)
        else:
            light_bbox = light_info[0]
            class_id = light_info[2]

            # Red Light
            if class_id == RED:
                self.data_buffer_traffic.append(RED)
            # Green Light
            elif class_id == GREEN:
                self.data_buffer_traffic.append(GREEN)

        self.idx = self.idx + 1 if self.idx < 10 else 0
        if self.idx == 0:
            most_class = max(
                set(self.data_buffer_traffic), key=self.data_buffer_traffic.count
            )

            if (
                self.data_buffer_traffic.count(most_class)
                / len(self.data_buffer_traffic)
                >= 0.7
            ):
                return most_class
        else:
            return COLLECTING

    def set_mode(self, mode: int):
        if mode < 0 and mode > 3:
            assert "Mode mode is a value between 0 and 2."
        self.cur_mode = mode

    def get_mode(self):
        return self.cur_mode

    def filter_info(self, detect_info_npy: np.ndarray) -> list:
        # Filtering : if Each zebra crossing(Stripe), traffic light have two pieces of information, reduce it to one
        class_ids = detect_info_npy[:, 5]
        stripes = detect_info_npy[class_ids == 0]
        light_red = detect_info_npy[class_ids == 1]
        light_green = detect_info_npy[class_ids == 2]

        # Stripes
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

    # Get a line from a point and unit vectors
    def lineCalc(self, vx, vy, x0, y0):
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

    # Vanishing point - cramer's rule
    def lineIntersect(self, m1, b1, m2, b2):
        # convert to cramer's system

        # a1*x + b1*y = c1
        # a2*x + b2*y = c2

        a1, b1, c1 = -m1, 1, b1
        a2, b2, c2 = -m2, 1, b2

        det = a1 * b2 - a2 * b1  # Determinant
        dx = c1 * b2 - c2 * b1
        dy = a1 * c2 - a2 * c2

        return dx / det, dy / det

    # The angle at the vanishing point
    def angleCalc(self, pt1, pt2):
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]

        return np.degrees(np.arctan2(dy, dx))

    def l2_distance(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def boxPoints(self, rect):
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        base = box[0]
        coords = []

        for i, point in enumerate(box[1:]):
            coords.append([i + 1, self.l2_distance(base, point)])

        coords = sorted(coords, key=lambda x: x[1])
        left = (box[0] + box[coords[0][0]]) // 2
        right = (box[coords[1][0]] + box[coords[2][0]]) // 2

        return left, right

    def find_outliers(self, data, lower=35, upper=50):
        # Compute IQR
        q1, q3 = np.percentile(data, [lower, upper])
        iqr = q3 - q1

        # Find outlier
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = np.logical_or(data < lower_bound, data > upper_bound)

        return ~outliers

    def set_find_param(self, timer=0, mode=0, error_cnt=0):
        self.timer = timer
        self.find_mode = mode
        self.error_cnt = error_cnt
