# lab 2 part 1: overlap the opencv logo on a background

import cv2    
import numpy as np

# overlap the opencv logo on a landscape taking advantage of the empty png area from the png.
# the images must have the same size and type.

####### this version keeps a transparency when overlapping the images
# logo = cv2.imread('opencv.png')
# scene = cv2.imread('scene.png')

# alpha = 1

# beta = 0.5
# dst = cv2.addWeighted(logo, alpha, scene, beta, 0)

# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#######

# Load two images
background_image = cv2.imread('nms.jpg')
logo_image = cv2.imread('opencv.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = logo_image.shape
roi = background_image[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
logo_image_gray = cv2.cvtColor(logo_image,cv2.COLOR_BGR2GRAY)

# both versions from below work. otsu helps you by finding the threshold
# ret,mask = cv2.threshold(logo_image_gray, 150, 255, cv2.THRESH_BINARY_INV)
ret,mask =  cv2.threshold(logo_image_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
background_image_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
logo_image_fg = cv2.bitwise_and(logo_image,logo_image,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(background_image_bg,logo_image_fg)
background_image[0:rows, 0:cols ] = dst

cv2.imshow('res',background_image)
cv2.waitKey(0)
cv2.destroyAllWindows()