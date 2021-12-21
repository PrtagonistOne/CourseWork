import sys
import numpy as np
import os
from utils.photo_func import resize_image, get_data_array, input_finger_data
from constants.photo_constants import SAMPLE_AMOUNT, SCALE_PERCENT


def main():
    np.set_printoptions(threshold=sys.maxsize)
    print(os.getcwd())
    os.chdir('1Finger')
    _, dim = resize_image(SCALE_PERCENT, 'ThumbOpened_' + str(1) + '.jpg')

    open_count = 1
    close_count = 0
    c_counter = 0
    data1 = get_data_array(dim)
    for _ in range(1):
        for k in range(SAMPLE_AMOUNT):
            original_count = k
            if k > 0:
                k = c_counter + 1
            c_counter = k + 9
            close_count += 2
            if close_count % 2 == 0:
                os.chdir('../1Finger')
                resized_closed, _ = resize_image(SCALE_PERCENT, 'ThumbClosed_' + str(close_count) + '.jpg')
                input_finger_data(resized_closed, k, data1, dim)
                # print(resized_closed, 'Written Closed Thumb#', close_count)  # thumb Closed

                os.chdir('../2Finger')
                resized_closed, _ = resize_image(SCALE_PERCENT, 'IndexClosed_' + str(close_count) + '.jpg')
                input_finger_data(resized_closed, k + 1, data1, dim)
                # print(resized_closed, 'Written Closed Index#', close_count)  # Index Closed

                os.chdir('../3Finger')
                resized_closed, _ = resize_image(SCALE_PERCENT, 'MiddleClosed_' + str(close_count) + '.jpg')
                input_finger_data(resized_closed, k + 2, data1, dim)
                # print(resized_closed, 'Written Closed Middle#', close_count)  # Middle Closed

                os.chdir('../4Finger')
                resized_closed, _ = resize_image(SCALE_PERCENT, 'RingClosed_' + str(close_count) + '.jpg')
                input_finger_data(resized_closed, k + 3, data1, dim)
                # print(resized_closed, 'Written Closed Ring#', close_count)  # Ring Closed

                os.chdir('../5Finger')
                resized_closed, _ = resize_image(SCALE_PERCENT, 'PinkyClosed_' + str(close_count) + '.jpg')
                input_finger_data(resized_closed, k + 4, data1, dim)
                # print(resized_closed, 'Written Closed Pinky#', close_count)  # Pinky Closed

            if original_count > 0:
                open_count += 2
            if open_count % 2 == 1:
                os.chdir('../1Finger')
                resized_opened, _ = resize_image(SCALE_PERCENT, 'ThumbOpened_' + str(open_count) + '.jpg')
                input_finger_data(resized_opened, k + 5, data1, dim)
                # print(resized_opened, 'Written Opened Thumb#', open_count)  # thumb opened

                os.chdir('../2Finger')
                resized_opened, _ = resize_image(SCALE_PERCENT, 'IndexOpened_' + str(open_count) + '.jpg')
                input_finger_data(resized_opened, k + 6, data1, dim)
                # print(resized_opened, 'Written Opened Index#', open_count)  # index opened

                os.chdir('../3Finger')
                resized_opened, _ = resize_image(SCALE_PERCENT, 'MiddleOpened_' + str(open_count) + '.jpg')
                input_finger_data(resized_opened, k + 7, data1, dim)
                # print(resized_opened, 'Written Opened Middle#', open_count)  # middle opened

                os.chdir('../4Finger')
                resized_opened, _ = resize_image(SCALE_PERCENT, 'RingOpened_' + str(open_count) + '.jpg')
                input_finger_data(resized_opened, k + 8, data1, dim)
                # print(resized_opened, 'Written Opened Ring#', open_count)  # ring opened

                os.chdir('../5Finger')
                resized_opened, _ = resize_image(SCALE_PERCENT, 'PinkyOpened_' + str(open_count) + '.jpg')
                input_finger_data(resized_opened, k + 9, data1, dim)
                # print(resized_opened, 'Written Opened Pinky#', open_count)  # pinky opened

            # print(k, k + 1, k + 2, k + 3, k + 4, k + 5, k + 6, k + 7, k + 8, k + 9)
            if c_counter > 49:
                # print(data1)
                break
    return data1


if __name__ == '__main__':
    main()
