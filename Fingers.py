import os.path
import sys
import cv2
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
directory = "D:\\CroppedData\\1Finger"
os.chdir(directory)
im_opened = cv2.imread('ThumbOpened_' + str(1) + '.jpg', cv2.IMREAD_UNCHANGED)

scale_percent = 10
width = int(im_opened.shape[1] * scale_percent / 100)
height = int(im_opened.shape[0] * scale_percent / 100)

openCount = 1
closeCount = 0
cCounter = 0
originalCount = 0
data1 = np.array([[[0 for n in range(width)] for i in range(height)] for j in range(60)])
for c in range(1):
    for k in range(12):
        dim = (width, height)
        originalCount = k
        if k > 0:
            k = cCounter + 1
        cCounter = k + 9
        closeCount = closeCount + 2

        if closeCount % 2 == 0:
            #print(closeCount)
            directory = "D:\\CroppedData\\1Finger"
            os.chdir(directory)
            im_closed = cv2.imread('ThumbClosed_' + str(closeCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_closed = cv2.resize(im_closed, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_closed[i][j] > 0:
                        data1[k][i][j] = 1
                    if resized_closed[i][j] == 0:
                        data1[k][i][j] = 0
            #print(resized_closed, 'Written Closed Thumb#', closeCount)  # thumb Closed
            directory = "D:\\CroppedData\\2Finger"
            os.chdir(directory)
            im_closed = cv2.imread('IndexClosed_' + str(closeCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_closed = cv2.resize(im_closed, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_closed[i][j] > 0:
                        data1[k + 1][i][j] = 1
                    if resized_closed[i][j] == 0:
                        data1[k + 1][i][j] = 0
            #print(resized_closed, 'Written Closed Index#', closeCount)  # Index Closed
            directory = "D:\\CroppedData\\3Finger"
            os.chdir(directory)
            im_closed = cv2.imread('MiddleClosed_' + str(closeCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_closed = cv2.resize(im_closed, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_closed[i][j] > 0:
                        data1[k + 2][i][j] = 1
                    if resized_closed[i][j] == 0:
                        data1[k + 2][i][j] = 0
            #print(resized_closed, 'Written Closed Middle#', closeCount)  # Middle Closed
            directory = "D:\\CroppedData\\4Finger"
            os.chdir(directory)
            im_closed = cv2.imread('RingClosed_' + str(closeCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_closed = cv2.resize(im_closed, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_closed[i][j] > 0:
                        data1[k + 3][i][j] = 1
                    if resized_closed[i][j] == 0:
                        data1[k + 3][i][j] = 0
            #print(resized_closed, 'Written Closed Ring#', closeCount)  # Ring Closed
            directory = "D:\\CroppedData\\5Finger"
            os.chdir(directory)
            im_closed = cv2.imread('PinkyClosed_' + str(closeCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_closed = cv2.resize(im_closed, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_closed[i][j] > 0:
                        data1[k + 4][i][j] = 1
                    if resized_closed[i][j] == 0:
                        data1[k + 4][i][j] = 0
            #print(resized_closed, 'Written Closed Pinky#', closeCount)  # Pinky Closed

        if originalCount > 0:
            openCount = openCount + 2
        if openCount % 2 == 1:
            directory = "D:\\CroppedData\\1Finger"
            os.chdir(directory)
            im_opened = cv2.imread('ThumbOpened_' + str(openCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_opened = cv2.resize(im_opened, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_opened[i][j] > 0:
                        data1[k + 5][i][j] = 1
                    if resized_opened[i][j] == 0:
                        data1[k + 5][i][j] = 0

            #print(resized_opened, 'Written Opened Thumb#', openCount)  # thumb opened
            directory = "D:\\CroppedData\\2Finger"
            os.chdir(directory)
            im_opened = cv2.imread('IndexOpened_' + str(openCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_opened = cv2.resize(im_opened, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_opened[i][j] > 0:
                        data1[k + 6][i][j] = 1
                    if resized_opened[i][j] == 0:
                        data1[k + 6][i][j] = 0

            #print(resized_opened, 'Written Opened Index#', openCount)  # index opened
            directory = "D:\\CroppedData\\3Finger"
            os.chdir(directory)
            im_opened = cv2.imread('MiddleOpened_' + str(openCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_opened = cv2.resize(im_opened, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_opened[i][j] > 0:
                        data1[k + 7][i][j] = 1
                    if resized_opened[i][j] == 0:
                        data1[k + 7][i][j] = 0

            #print(resized_opened, 'Written Opened Middle#', openCount)  # middle opened
            directory = "D:\\CroppedData\\4Finger"
            os.chdir(directory)
            im_opened = cv2.imread('RingOpened_' + str(openCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_opened = cv2.resize(im_opened, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_opened[i][j] > 0:
                        data1[k + 8][i][j] = 1
                    if resized_opened[i][j] == 0:
                        data1[k + 8][i][j] = 0

            #print(resized_opened, 'Written Opened Ring#', openCount)  # ring opened
            directory = "D:\\CroppedData\\5Finger"
            os.chdir(directory)
            im_opened = cv2.imread('PinkyOpened_' + str(openCount) + '.jpg', cv2.IMREAD_UNCHANGED)
            resized_opened = cv2.resize(im_opened, dim, interpolation=cv2.INTER_AREA)
            for i in range(height):
                for j in range(width):
                    if resized_opened[i][j] > 0:
                        data1[k + 9][i][j] = 1
                    if resized_opened[i][j] == 0:
                        data1[k + 9][i][j] = 0

            #print(resized_opened, 'Written Opened Pinky#', openCount)  # pinky opened


        print(k, k+1, k+2, k+3, k+4, k+5, k+6, k+7, k+8, k+9)
        if cCounter > 49:
            #print(data1)
            break


