# lab 3: extract the corners on 2 images

import cv2
import numpy as np


def extract_corners():
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    diamond_kernel = np.array([[0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 0],
                               [0, 0, 1, 0, 0]], dtype=np.uint8)
    x_shape_kernel = np.array([[1, 0, 0, 0, 1],
                               [0, 1, 0, 1, 0],
                               [0, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0],
                               [1, 0, 0, 0, 1]], dtype=np.uint8)
    square_kernel = np.ones((5, 5), dtype=np.uint8)

    o_img = cv2.imread("building.png")
    img = cv2.cvtColor(o_img, cv2.COLOR_BGR2GRAY)

    #  determines the threshold automatically from the image using Otsu's method,
    # (thresh, im_bw) = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY_INV)

    # 1. R1 = Dilate(Img,cross)
    dilated_im = cv2.dilate(img, kernel=cross_kernel)

    # 2. R1 = Erode(R1,Diamond)
    r1 = cv2.erode(dilated_im, kernel=diamond_kernel)

    # 3. R2 = Dilate(Img,Xshape)
    dilated_im = cv2.dilate(img, kernel=x_shape_kernel)

    # 4. R2 = Erode(R2,square)
    r2 = cv2.erode(dilated_im, kernel=square_kernel)

    # 5. R = absdiff(R2,R1)
    abs_diff = cv2.absdiff(r2, r1)

    tophat = cv2.morphologyEx(
        abs_diff, cv2.MORPH_TOPHAT, kernel=np.ones((3, 3), dtype=np.uint8))

    (thresh, res) = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY_INV)

    rows, cols = res.shape
    for row in range(rows):
        for col in range(cols):
            if(res[row][col] == 0):
                cv2.circle(o_img, (col, row), 2, (255, 0, 0), thickness=1)

    intersection = cv2.bitwise_and(img, res)

    cv2.imshow('img', o_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    extract_corners()
