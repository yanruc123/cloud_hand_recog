import cv2
import numpy as np
import math
import imutils
import time
import os
import random


def calculateAngle(far, start, end):
    """Cosine rule"""
    temp_a1 = end[0] - start[0]
    temp_a2 = end[1] - start[1]
    a = math.sqrt(temp_a1 ** 2 + temp_a2 ** 2)
    temp_b1 = far[0] - start[0]
    temp_b2 = far[1] - start[1]
    b = math.sqrt(temp_b1 ** 2 + temp_b2 ** 2)
    temp_c1 = end[0] - far[0]
    temp_c2 = end[1] - far[1]
    c = math.sqrt(temp_c1 ** 2 + temp_c2 ** 2)
    angle = math.acos((b ** 2 - a ** 2 + c ** 2) / (2 * c * b))
    return angle * 57


def recognize(frame):
    img = frame.copy()
    cv2.rectangle(frame, (26, 0), (340, 550), (170, 170, 0))
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(grey_img, 80, 255, cv2.THRESH_BINARY_INV)
    #print("here2")

    # 利用BackgroundSubtractorMOG2算法消除背景
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(binary)

    # 去噪
    fgmask = cv2.erode(binary, (3, 3))  # 腐蚀操作
    fgmask = cv2.dilate(binary, (3, 3), iterations=1)  # 膨胀操作
    res = cv2.bitwise_and(binary, binary, mask=fgmask)
    skin = cv2.GaussianBlur(res, (11, 11), 0)  # 高斯滤波
    #cv2.imshow("skin", skin)

    ret, inverse_skin = cv2.threshold(skin, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("inverse_skin", inverse_skin)

    cnts, h = cv2.findContours(inverse_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

    cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)

    # Find convex hull and defects
    largecont = max(cnts, key=lambda contour: cv2.contourArea(contour))
    hull2 = cv2.convexHull(largecont, returnPoints=False)  # For the indices
    defects = cv2.convexityDefects(largecont, hull2)

    #print("defects = ", defects)

    # Draw defect points
    count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largecont[s][0])
            end = tuple(largecont[e][0])
            far = tuple(largecont[f][0])
            angle = calculateAngle(far, start, end)
            # Ignore the defects which are small and wide Probably not fingers
            if d > 50 and 75 >= angle >= 25:
                count += 1
                cv2.circle(frame, far, 3, [255, 0, 0], -1)
            cv2.line(frame, start, end, [0, 255, 0], 2)

    else:
        return "Invalid"

    result = 0
    if count == 1 or count == 2:
        cv2.putText(frame, "Scissors", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        result = "Scissors"
    elif count == 0:
        cv2.putText(frame, "Rock", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        result = "Rock"
    elif count >= 3:
        cv2.putText(frame, "Paper", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        result = "Paper"
    else:
        cv2.putText(frame, "invalid hand gesture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                    cv2.LINE_AA)

    cv2.imshow("skin", skin)
    cv2.imshow("frame", frame)

    return result


def game():
    print("this is a rock-paper-scissor game using vidcam")
    print("press enter key to start")
    print("put your hand as close to the box as possible")

    input("press enter to continue")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened:
        print('Unable to open camera')
        exit(0)

    start_time = time.time()
    print("you have 20 sec to choose:")

    while True:
        # grab the current frame
        (grabbed, frame) = capture.read()
        cv2.imshow("frame", frame)
        #print("here1")

        user_give = recognize(frame)
        #recognize(frame)
        #print(user_give)
        end_time = time.time()
        cv2.putText(frame, str(int((5 - (end_time - start_time)))), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        #cv2.imshow("frame", frame)

        if end_time - start_time > 20:
            break

        if cv2.waitKey(30) >= 0:
            break

    ha, img = capture.read()
    capture.release()
    cv2.imshow("camera", img)

    computer_choice = ["Rock", "Paper", "Scissor"]
    computer_give = random.choice(computer_choice)

    print("user:", user_give)
    print("computer: ",computer_give)

    # judge
    judgement = judge(user_give, computer_give)
    if judgement == 0:
        print("user win")
    elif judgement == 1:
        print("computer win")
    elif judgement == 2:
        print("draw")
    else:
        print("Please put your hand closer to the camera or choose a better spot")


def judge(u, c):
    """
    0:user win
    1:computer win
    2:draw
    """
    if u == "Rock":
        if c == "Scissor":
            return 1
        if c == "Rock":
            return 2
        elif c == "Paper":
            return 0
    elif u == "Scissor":
        if c == "Scissor":
            return 2
        if c == "Rock":
            return 1
        elif c == "Paper":
            return 0
    else:  # u == "Paper"
        if c == "Scissor":
            return 1
        if c == "Rock":
            return 0
        elif c == "Paper":
            return 2


game()

# https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
# https://blog.csdn.net/weixin_44885615/article/details/97811684
