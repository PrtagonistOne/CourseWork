# organize imports
import math
import os.path
import random
import sys
import cv2
import imutils
import numpy as np
from Fingers import data1
from DeterminedResults import corrRes
from google.protobuf.json_format import MessageToDict
import mediapipe as mp

np.set_printoptions(threshold=sys.maxsize)
mp_hands = mp.solutions.hands


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


class Neurons_Hidden:
    def __init__(self, speed, dim):
        self.weights = np.array(
            [[[-round(random.uniform(-0.5, 0.5), 6)] for i in range(dim[0])] for j in range(dim[1])])
        self.speed = speed
        self.b = -round(random.uniform(-0.5, 0.5), 6)

    def Activate(self, x):
        S = 0
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                S += x[i][j] * self.weights[i][j]

        S = S + self.b
        return 1.0 / (1.0 + math.exp(-S))

    def WeightsAdjustment(self, err, y):
        self.b = self.b + self.speed * err
        row, col = y.shape
        for i in range(row):
            for j in range(col):
                self.weights[i][j] += self.speed * err * y[i][j]


class Neurons_Output:
    def __init__(self, weights, speed):
        self.weights = np.array([[-round(random.uniform(-0.5, 0.5), 6)] for i in range(weights)])
        self.speed = speed
        self.b = -round(random.uniform(-0.5, 0.5), 6)

    def Activate_Out(self, x):
        S = 0
        for i in range(len(x)):
            S += x[i] * self.weights[i]
        S = S + self.b
        return 1.0 / (1.0 + math.exp(-S))

    def WeightsAdjustment_Out(self, err, y):
        self.b = self.b + self.speed * err
        for i in range(len(self.weights)):
            self.weights[i] += self.speed * err * y[i]


class Perceptron:
    def __init__(self, amount, speed, data, corrRes, dim):
        self.HiddenNeurons = []
        self.Output = []
        self.data = data
        self.corrRes = corrRes
        self.dim = dim
        self.amount = amount
        for i in range(10):
            self.Output.append(Neurons_Output(self.amount, speed))
        for i in range(self.amount):
            self.HiddenNeurons.append(Neurons_Hidden(speed, dim))

    def Check(self, inputData):
        results = np.zeros(self.amount, dtype=float)
        for i in range(len(results)):
            results[i] = self.HiddenNeurons[i].Activate(inputData)

        finalRes = np.zeros(10, dtype=float)
        for i in range(10):
            finalRes[i] = self.Output[i].Activate_Out(results)

        return finalRes

    def Learn(self, epoch):
        z, y, x = self.data.shape
        while epoch != 0:
            for k in range(z):
                hidRes = np.zeros(z, dtype=float)
                outRes = np.zeros(10, dtype=float)
                outDelta = np.zeros(10, dtype=float)
                for i in range(len(self.HiddenNeurons)):
                    hidRes[i] = self.HiddenNeurons[i].Activate(self.data[k])

                for i in range(len(self.Output)):
                    outRes[i] = self.Output[i].Activate_Out(hidRes)

                for i in range(len(self.Output)):
                    outDelta[i] = outRes[i] * (1 - outRes[i]) * (self.corrRes[i][k] - outRes[i])
                    self.Output[i].WeightsAdjustment_Out(outDelta[i], hidRes)

                for i in range(len(self.HiddenNeurons)):
                    hidDelta = 0
                    for j in range(len(self.Output)):
                        hidDelta += outDelta[j] * self.Output[j].weights[i]

                    hidError = hidRes[i] * (1 - hidRes[i]) * hidDelta
                    self.HiddenNeurons[i].WeightsAdjustment(hidError, self.data[k])

            print("Epoch#", epoch)
            epoch -= 1


aWeight = 0.5
camera = cv2.VideoCapture(0)
top, right, bottom, left = 60, 430, 245, 615
num_frames = 0
bg = None
count = 0
while True:

    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (height, width) = frame.shape[:2]

    roi = frame[top:bottom, right:left]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        hand = segment(gray)
        if hand is not None:
            (thresholded, segmented) = hand
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
            cv2.imshow("Threshold", thresholded)

    directory = "D:\\CroppedData\\5Finger"
    os.chdir(directory)
    im_opened = cv2.imread('PinkyOpened_' + str(1) + '.jpg', cv2.IMREAD_UNCHANGED)

    scale_percent = 10
    width = int(im_opened.shape[1] * scale_percent / 100)
    height = int(im_opened.shape[0] * scale_percent / 100)
    dim = (width, height)

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
    num_frames += 1
    cv2.imshow("Video Feed", clone)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break
    if keypress == ord('s'):
        count += 1
        directory = "D:\\CroppedData"
        os.chdir(directory)
        (width, height) = dim

        cv2.imwrite('HandType' + str(count) + '.jpg', frame)
        image = cv2.imread('HandType' + str(count) + '.jpg', cv2.IMREAD_UNCHANGED)
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for idx, handType in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(handType)
                typeH = handedness_dict['classification'].__getitem__(0)

        if typeH['label'] == 'Left':
            cv2.imwrite('Test' + str(count) + '.jpg', cv2.flip(thresholded, 1))
        if typeH['label'] == 'Right':
            cv2.imwrite('Test' + str(count) + '.jpg', thresholded)

        cv2.imwrite('Test' + str(count) + '.jpg', thresholded)
    if keypress == ord('l'):
        # Навчання нейроної мережі

        data = data1
        print(corrRes)
        Per = Perceptron(60, 0.4, data, corrRes, dim)
        Per.Learn(60)
    if keypress == ord('1'):
        for k in range(1):
            directory = "D:\\CroppedData"
            os.chdir(directory)
            print(count)
            im = cv2.imread('Test' + str(count) + '.jpg', cv2.IMREAD_UNCHANGED)
            resizedImg = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

            check = np.array([[0 for i in range(width)] for i in range(height)])
            for i in range(height):
                for j in range(width):
                    if resizedImg[i][j] > 0:
                        check[i][j] = 1
                    if resizedImg[i][j] == 0:
                        check[i][j] = 0
            res = Per.Check(check)

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

            if typeH['label'] == 'Right':
                cv2.putText(frame,
                            'Right hand',
                            (50, 300),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)
            if typeH['label'] == 'Left':
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
