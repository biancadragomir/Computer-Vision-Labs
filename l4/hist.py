# lab 4: histogram stuff

import cv2
import numpy as np
import matplotlib.pyplot as plt

typesOfImages = ['people', 'dogs', 'cars', 'landscapes']


def computeHistogramForCategory(categoryName):
    images = []
    for currentImageIndex in range(8):
        currFilename = categoryName + "/" + str(currentImageIndex) + ".jpg"
        curr_img = cv2.imread(currFilename)
        # print("Processing image: " + currFilename)
        hsv_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        images.append(hsv_img)

    hist = cv2.calcHist(images, channels=[0], mask=None,
                        histSize=[256], ranges=[0, 256])
    hist = cv2.normalize(hist, hist, alpha=0,
                         beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


testFiles = ["test_person.jpg", "test_dog2.jpg",
             "test_car.jpg", "test_landscape.jpg"]


def run(histMetric, currentCategory):

    # https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    if(histMetric == cv2.HISTCMP_INTERSECT):
        print("\n\nCurrent histogram metric is intersection")
    if(histMetric == cv2.HISTCMP_BHATTACHARYYA):
        print("\n\nCurrent histogram metric is HISTCMP_BHATTACHARYYA")
    if(histMetric == cv2.HISTCMP_CHISQR):
        print("\n\nCurrent histogram metric is HISTCMP_CHISQR")
    if(histMetric == cv2.HISTCMP_CORREL):
        print("\n\nCurrent histogram metric is Correlation")

    testImage = cv2.imread(testFiles[currentCategory])

    print("Current test photo is "+testFiles[currentCategory])

    hsvTestImg = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
    testHist = cv2.calcHist(hsvTestImg, channels=[0], mask=None,
                            histSize=[256], ranges=[0, 256])
    testHist = cv2.normalize(testHist, testHist, alpha=0,
                             beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    histograms = []
    currMax = -1
    maxCateg = ""
    maxHist = None

    for category in typesOfImages:
        # print("\n" + "Current category: " + category)
        currentHistogram = computeHistogramForCategory(category)
        histograms.append(currentHistogram)

        comparison = cv2.compareHist(
            currentHistogram, testHist, histMetric)
        if(currMax < comparison):
            currMax = comparison
            maxCateg = category
            maxHist = currentHistogram

        print("Image compared to " + category+": " +
              str(comparison))
    print("Max is "+str(currMax) + " for category "+maxCateg)

    if (maxCateg == typesOfImages[currentCategory]):
        print("---------------------- GOT IT RIGHT ----------------------------")
    else:
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    currCategory = 0

    # uncomment for plotting the results
    # plt.plot(histogram)
    # plt.title("histogram")
    # plt.show()


if __name__ == "__main__":

    print("0 - person\n1 - dog\n2 - car\n3 - landscape")
    # set here the test type
    currentCategory = int(input(("Enter the desired test photo: ")))

    testedHistMetric = cv2.HISTCMP_INTERSECT
    run(testedHistMetric, currentCategory)

    testedHistMetric = cv2.HISTCMP_BHATTACHARYYA
    run(testedHistMetric, currentCategory)

    testedHistMetric = cv2.HISTCMP_CHISQR
    run(testedHistMetric, currentCategory)

    testedHistMetric = cv2.HISTCMP_CORREL
    run(testedHistMetric, currentCategory)
