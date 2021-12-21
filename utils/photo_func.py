import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict
import mediapipe as mp

bg = None


def run_avg(image, a_weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, a_weight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    segmented = max(cnts, key=cv2.contourArea)
    return thresholded, segmented


def resize_image(scale_percent: int, img_path: str) -> tuple:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # print('Original Dimensions : ', img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print('Resized Dimensions : ', resized.shape)
    return resized, dim


def get_data_array(dim: tuple) -> np.array:
    width, height = dim
    return np.array(
        [[[0 for n in range(width)] for i in range(height)] for j in range(60)]
    )


def input_finger_data(img: list, k: int, data: np.array, dim: tuple) -> None:
    width, height = dim

    for i in range(height):
        for j in range(width):
            if img[i][j] > 0:
                data[k][i][j] = 1
            if img[i][j] == 0:
                data[k][i][j] = 0


def take_screenshot(count: int, thresholded: list, frame: cv2.flip) -> dict:
    mp_hands = mp.solutions.hands

    cv2.imwrite('HandType' + str(count) + '.jpg', frame)
    image = cv2.imread('HandType' + str(count) + '.jpg', cv2.IMREAD_UNCHANGED)
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for handType in results.multi_handedness:
            handedness_dict = MessageToDict(handType)
            type_hand = handedness_dict['classification'].__getitem__(0)

    if type_hand['label'] == 'Left':
        cv2.imwrite('Test' + str(count) + '.jpg', cv2.flip(thresholded, 1))
    if type_hand['label'] == 'Right':
        cv2.imwrite('Test' + str(count) + '.jpg', thresholded)

    cv2.imwrite('Test' + str(count) + '.jpg', thresholded)

    return type_hand
