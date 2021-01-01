import sys
import math
import cv2 as cv
import numpy as np

# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html


def detect():

    src = cv.imread("sudoku.png", cv.IMREAD_GRAYSCALE)

    # Edge detection using a canny detector
    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    '''
    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    lines: A vector that will store the parameters (r,θ) of the detected lines
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    srn and s`tn: Default parameters to zero. Check OpenCV reference for more info.
    '''
    # keep the threshold between 150 and 200. otherwise the detection is exaggerated!
    lines = cv.HoughLines(dst, 1, np.pi / 180, 250)

    # Here we draw the lines that result from the hough algorithm
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            cv.line(cdst, pt1, pt2, (200, 35, 100), 3, cv.LINE_AA)

    # for the circles: we apply medianBlur to reduce the noise to avoid false circle detection
    gray = cv.medianBlur(src, 5)

    rows = gray.shape[0]

    '''
    param_1 = 200: Upper threshold for the internal Canny edge detector.
    param_2 = 100*: Threshold for center detection.
    '''
    # played around with maxRadius. min 20 - detects stuff. increase this value to detect more circles!
    # changing param_1 does nothing.
    # changing param_2 to 10 messes up the centers. they are waaay off.
    # 30 seems fine. 50 messes up some circles on an "8"
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=10, maxRadius=50)

    # Here we draw the circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(cdst, center, radius, (200, 30, 255), 3)

    cv.imshow("lines & circles", cdst)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    detect()
