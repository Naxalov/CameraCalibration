import os
import cv2
import numpy as np

# Read image from folder

PATH = os.path.join(os.getcwd(), 'circle')
images = os.listdir(PATH)
img_path = os.path.join(PATH, images[3])
# Loads an image
img = cv2.imread(img_path)

# get shape of image
W, H, _ = img.shape

# Check if image is loaded fine
if img is None:
    print('Error opening image!')
# image resize
img = cv2.resize(img, (H // 4, W // 4))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
output = img

# detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, W // 16)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=15, param1=50, param2=18, minRadius=6,
                           maxRadius=32)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image
    cv2.imshow("output", np.hstack([img, output]))
    cv2.waitKey(0)
