import os
import cv2 as cv
import face_recognition
import numpy as np
from speaker import speak
from threading import Thread
import time
import pytesseract
from PIL import Image

main_frame = []
frame_num = 0
face_start_key = False
finish = False
count = 0
regist = False
read = False
unknown_count = 0


def face_recog():
    global frame_num
    global main_frame
    global face_start_key
    global finish
    global regist
    global read
    global unknown_count

    # cap = cv.VideoCapture(0) # webcam 사용
    # cap = cv.videoCapture(1) # Camera use
    # cap = cv.VideoCapture(gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)

    img_names = []
    img_list = []
    known_face_encodings = []
    global count
    known_face_names = []

    # 이미지 이름 배열
    print("부팅 시작...!!!")

    for i in range(50):
        img_names.append("face_recognition/" + str(i + 1) + ".jpg")

    # img_list 로 폴더에 존재하는 이미지 로드.
    for i in img_names:
        try:
            img_list.append(face_recognition.load_image_file(str(i)))
            count = count + 1
        except:
            break
    print(str(count) + "개의 이미지 로드 됨")

    # known_face_encodings 에 인코딩 결과 저장
    for i in img_list:
        known_face_encodings.append(face_recognition.face_encodings(i)[0])

    # 이름 저장된 txt파일에서 이름 목록 읽어오기
    f_names = open("face_recognition/names.txt", "r")
    print(f_names)
    lines = f_names.readlines()
    for line in lines:
        known_face_names.append(line.strip())
    f_names.close()

    # 변수 초기화
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    centerX = 0
    centerY = 0
    recognized_name = {}
    unknown_count = 0
    # cv.imshow('face_recognition', main_frame)

    # print("Camera on")
    # ret, img_color = cap.read()
    while True:
        if finish:
            break
        # ret, img_color = cap.read()
        img_color = main_frame.copy()
        # key = cv.waitKey(1)

        # r 키 누르면 등록
        if regist:
            img_for_capture = cv.resize(img_color, (0, 0), fx=1.0, fy=1.0)
            img_capture = cv.imwrite(
                "face_recognition/" + str(count + 1) + ".jpg", img_for_capture
            )
            count = count + 1
            face_locations = face_recognition.face_locations(
                face_recognition.load_image_file(
                    "face_recognition/" + str(count) + ".jpg"
                )
            )
            if len(face_locations) != 1:
                print(
                    "I found {} face(s) in this photograph. so capture canceled.".format(
                        len(face_locations)
                    )
                )
                count = count - 1
                os.remove("face_recognition/" + str(count + 1) + ".jpg")
            else:
                f = open("face_recognition/names.txt", "a")
                data = input("이름 : ")
                known_face_names.append(str(data))
                img_list.append(
                    face_recognition.load_image_file(
                        "face_recognition/" + str(count) + ".jpg"
                    )
                )
                known_face_encodings.append(
                    face_recognition.face_encodings(img_list[count - 1])[0]
                )
                data = str(data) + "\n"
                f.write(data)
            regist = False

        # t key => capture for read
        if read:
            img_for_capture = cv.resize(img_color, (0, 0), fx=1.0, fy=1.0)
            img_capture = cv.imwrite("face_recognition/text.jpg", img_for_capture)

            image = Image.open("face_recognition/text.jpg")
            text = pytesseract.image_to_string(image)
            print(text)
            speak(text)
            read = False

        if process_this_frame == 15:
            process_this_frame = 0
            img_color_for_encoding = cv.resize(img_color, (0, 0), fx=0.5, fy=0.5)
            img_for_capture = cv.resize(img_color, (0, 0), fx=1.0, fy=1.0)
            rgb_img = img_color_for_encoding[:, :, ::-1]
            rgb_img = np.array(rgb_img)
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            """
            centerX, centerY = 0, 0
            if face_locations:
                centerX = ((face_locations[0][1]) + (face_locations[0][3])) / 2
                centerY = ((face_locations[0][0]) + (face_locations[0][2])) / 2
            """

            face_names = []
            for face_encoding in face_encodings:
                #    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                #    if True in matches:
                #        first_match_index = matches.index(True)
                #        name = known_face_names[first_match_index]
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                # print(face_distances)

                if face_distances[best_match_index] < 0.4:
                    name = known_face_names[best_match_index]

                if name == "Unknown":
                    unknown_count += 1
                    if unknown_count >= 10:
                        print("Unknown person")
                        speak("Caution. Uknown person.")
                        unknown_count = 0

                elif name not in recognized_name:
                    speak(name)
                    print("1")
                    recognized_name[name] = frame_num
                    unknown_count = 0
                elif frame_num - recognized_name[name] > 500:
                    speak(name)
                    print("2")
                    recognized_name[name] = frame_num
                    unknown_count = 0

                face_names.append(name)

                # if centerY != 0 and centerX != 0:
                #     if centerX < width - (cell_width * 2):
                #         print("{} is on leftX".format(name))
                #         speak( name + " is on leftX")
                #         left_flag = 1
                #         right_flag = 0
                #     elif centerX > width - cell_width:
                #         print("{} is on rightX".format(name))
                #         speak(name+" is on  rightX")
                #         right_flag = 1
                #         left_flag = 0
                #     else:
                #         left_flag = 0
                #     if centerY < height - (cell_height * 2):
                #         print("{} is on topY".format(name))
                #         speak(name+" is on  topY")
                #         top_flag = 1
                #         bottom_flag = 0
                #     elif centerY > height - cell_height:
                #         print("{} is on bottomY".format(name))
                #         speak(name+" is on  bottomY")
                #         top_flag = 0
                #         bottom_flag = 1
                #     else:
                #         print("{} is infront of you".format(name))
                #         speak(name+" is  infront of you")
                #         top_flag = 0
                #         bottom_flag = 0

        process_this_frame = process_this_frame + 1
        # if process_this_frame < 100:
        #    process_this_frame = process_this_frame + 1
        # else:
        #    process_this_frame = 0

        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 인코딩에 사용한 이미지가 1/16사이즈였으므로 원본이미지상에서 얼굴위치 표시 시, 4배해주어야 함
        #     top *= 2
        #    right *= 2
        #    bottom *= 2
        #    left *= 2

        # 얼굴 Box 그리기
        #    cv.rectangle(img_color, (left, top), (right, bottom), (0, 0, 255), 2)

        # # 이름 쓰기
        #    cv.rectangle(img_color, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        #    font = cv.FONT_HERSHEY_DUPLEX
        #    cv.putText(img_color, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #    speak(name)

        # cv.imshow('face_recognition', img_color)

    # cap.release()
    cv.destroyAllWindows()


def streaming():
    global frame_num
    global face_start_key
    global main_frame
    global finish
    global count
    global known_face_names
    global regist
    global read
    global uknown_count

    cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)

    if not cap.isOpened():
        print("camera is not openned")
        return 0
    ret, main_frame = cap.read()
    face_start_key = True

    while ret:
        cv.imshow("streaming", main_frame)
        ret, main_frame = cap.read()
        frame_num += 1
        key = cv.waitKey(1)

        if key & 0xFF == 27:
            finish = True
            break
        elif key & 0xFF == 114:
            regist = True
        elif key & 0xFF == 116:
            read = True

        # r 키 누르면 등록
        elif key & 0xFF == 114:
            img_color = main_frame.copy()
            img_for_capture = cv.resize(img_color, (0, 0), fx=1.0, fy=1.0)
            img_capture = cv.imwrite(
                "face_recognition/" + str(count + 1) + ".jpg", img_for_capture
            )
            count = count + 1
            face_locations = face_recognition.face_locations(
                face_recognition.load_image_file(
                    "face_recognition/" + str(count) + ".jpg"
                )
            )
            # 얼굴이 없거나, 얼굴이 여러 개
            if len(face_locations) != 1:
                print(
                    "I found {} face(s) in this photograph. so capture canceled.".format(
                        len(face_locations)
                    )
                )
                count = count - 1
                os.remove("face_recognition/" + str(count + 1) + ".jpg")
            else:
                f = open("face_recognition/names.txt", "a")
                data = input("이름 : ")
                known_face_names.append(str(data))
                img_list.append(
                    face_recognition.load_image_file(
                        "face_recognition/" + str(count) + ".jpg"
                    )
                )
                known_face_encodings.append(
                    face_recognition.face_encodings(img_list[count - 1])[0]
                )
                data = str(data) + "\n"
                f.write(data)


def main():
    frame_num = 0
    face_start_key = False

    th1 = Thread(target=face_recog, args=())
    th2 = Thread(target=streaming)
    th2.start()
    th1.start()

    # while ret:
    # 	cv.imshow('streaming', main_frame)
    # 	ret, main_frame = cap.read()
    # 	frame_num+=1
    # 	key = cv.waitKey(1)
    # 	if key & 0xFF == 27:
    # 		break

    th1.join()
    th2.join()
    # cap.release()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
