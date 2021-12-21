import os.path
import sys
import cv2
import imutils
import numpy as np
from utils.Fingers import main as data1
from constants.DeterminedResults import corrRes
from utils.neural_network import Perceptron
from utils.photo_func import run_avg, segment, resize_image, take_screenshot
from constants.photo_constants import SCALE_PERCENT
from constants.perceptron_constants import EPOCHS, NEURONS_AMOUNT
from constants.videocamera_constants import TOP, RIGHT, BOTTOM, LEFT, NUM_FRAMES, SCREENSHOT_COUNT, A_WEIGHT
np.set_printoptions(threshold=sys.maxsize)

camera = cv2.VideoCapture(0)
_, dim = resize_image(SCALE_PERCENT, 'CroppedData/5Finger/PinkyOpened_' + str(1) + '.jpg')
os.chdir('CroppedData')
while True:

    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    roi = frame[TOP:BOTTOM, RIGHT:LEFT]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if NUM_FRAMES < 30:
        run_avg(gray, A_WEIGHT)
    else:
        hand = segment(gray)
        if hand is not None:
            (thresholded, segmented) = hand
            cv2.drawContours(clone, [segmented + (RIGHT, TOP)], -1, (0, 0, 255))
            cv2.imshow("Threshold", thresholded)

    cv2.rectangle(clone, (LEFT, TOP), (RIGHT, BOTTOM), (0, 255, 0), 2)
    NUM_FRAMES += 1
    cv2.imshow("Video Feed", clone)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break
    if keypress == ord('s'):

        SCREENSHOT_COUNT += 1
        type_hand = take_screenshot(SCREENSHOT_COUNT, thresholded, frame)

    if keypress == ord('l'):
        # Навчання нейроної мережі
        data = data1()
        Per = Perceptron(NEURONS_AMOUNT, 0.45, data, corrRes, dim)
        Per.learn(EPOCHS)
    if keypress == ord('1'):
        (width, height) = dim
        for _ in range(1):
            resizedImg, _ = resize_image(SCALE_PERCENT, 'Test' + str(SCREENSHOT_COUNT) + '.jpg')
            check = np.array([[0 for i in range(width)] for j in range(height)])
            for i in range(height):
                for j in range(width):
                    if resizedImg[i][j] > 0:
                        check[i][j] = 1
                    if resizedImg[i][j] == 0:
                        check[i][j] = 0
            res = Per.check(check)

            print(res)

            mes1 = "Thumb is closed! With " + str(round(res[0] * 100, 2)) + "% "
            mes6 = "Thumb is opened! With " + str(round(res[5] * 100, 2)) + "% "

            mes2 = "Index finger is closed! With " + str(round(res[1] * 100, 2)) + "% "
            mes = "Index finger is opened! With " + str(round(res[6] * 100, 2)) + "% "

            mes3 = "Middle finger is closed! With " + str(round(res[2] * 100, 2)) + "% "
            mes8 = "Middle finger is opened! With " + str(round(res[7] * 100, 2)) + "% "

            mes4 = "Ring finger is closed! With " + str(round(res[3] * 100, 2)) + "% "
            mes9 = "Ring finger is opened! With " + str(round(res[8] * 100, 2)) + "% "

            mes5 = "Pinky is closed! With " + str(round(res[4] * 100, 2)) + "% "
            mes7 = "Pinky is opened! With " + str(round(res[9] * 100, 2)) + "% "

            if res[0] > res[5]:
                cv2.putText(frame,
                            mes1,
                            (50, 50),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if res[0] < res[5]:
                cv2.putText(frame,
                            mes6,
                            (50, 50),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)

            if res[1] > res[6]:
                cv2.putText(frame,
                            mes2,
                            (50, 80),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if res[1] < res[6]:
                cv2.putText(frame,
                            mes,
                            (50, 80),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)

            if res[2] > res[7]:
                cv2.putText(frame,
                            mes3,
                            (50, 110),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if res[2] < res[7]:
                cv2.putText(frame,
                            mes8,
                            (50, 110),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)

            if res[3] > res[8]:
                cv2.putText(frame,
                            mes4,
                            (50, 140),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if res[3] < res[8]:
                cv2.putText(frame,
                            mes9,
                            (50, 140),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)

            if res[4] > res[9]:
                cv2.putText(frame,
                            mes5,
                            (50, 170),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if res[4] < res[9]:
                cv2.putText(frame,
                            mes7,
                            (50, 170),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)

            if type_hand['label'] == 'Right':
                cv2.putText(frame,
                            'Right hand',
                            (50, 300),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if type_hand['label'] == 'Left':
                cv2.putText(frame,
                            'Left hand',
                            (50, 300),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)
            cv2.imshow('Result', frame)

# free up memory
camera.release()
cv2.destroyAllWindows()
