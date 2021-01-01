# lab 2 part 2: flood fill using dilations

import cv2
import numpy as np

def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def flood_fill():
    img = cv2.imread("box.jpg", cv2.IMREAD_GRAYSCALE)

    #  determines the threshold automatically from the image using Otsu's method, 
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # dimensions of the image
    rows, cols = im_bw.shape
    empty_canvas = np.zeros((rows, cols), np.uint8)

    kernel = np.ones((3,3), np.uint8)

    # this will extract the contour of the shape
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_GRADIENT, kernel=kernel)

    starting_point_x = int(cols/2)
    starting_point_y = int(rows/2) 

    # STEP 1: set a starting point inside of the contour
    empty_canvas[starting_point_x, starting_point_y] = 255
    im_complementary = cv2.bitwise_not(im_bw)
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        # STEP 2 p1: dilate the current "final" result
        temp_im = cv2.dilate(empty_canvas, cross_kernel)
        # STEP 2 p2: intersect the dilated result with the complementary of our image
        temp_canvas = cv2.bitwise_and(temp_im, temp_im, mask=im_complementary)

        # STEP 3: if the new result is the same as the previous, then we covered all the interior points of the contour
        if is_similar(temp_canvas, empty_canvas):
            break
        # otherwise, we need to continue filling
        empty_canvas = temp_canvas
        
    # STEP 4: add the canvas to the original img
    final_result = cv2.add(empty_canvas, im_bw)

    cv2.imshow('img', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    flood_fill()