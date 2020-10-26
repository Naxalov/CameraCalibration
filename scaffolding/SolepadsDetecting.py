import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def get_center_points(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pad_center = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pad_center.append([cX, cY])
        print(cX, cY)
        # cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display the image
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
    return pad_center


# input image in order calibration
INPUT_DIR = 'solepads/square'
PATH = os.path.join(os.getcwd(), INPUT_DIR)

images = os.listdir(PATH)
img = cv2.imread(os.path.join(PATH, images[0]))
# get shape of image
W, H, _ = img.shape
img = cv2.resize(img, (H // 4, W // 4))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img, (11, 11), 0)
# RGB [13,15,12]
minBGR = np.array([4, 4, 4])
maxBGR = np.array([45, 45, 45])

mask = cv2.inRange(img, minBGR, maxBGR)
# result = cv2.bitwise_and(nemo, nemo, mask=mask)
pnt = get_center_points(mask)
for cor in pnt:
    cv2.circle(img_rgb, tuple(cor), 5, (255, 0, 0), -1)
    cv2.putText(img_rgb, "centroid", (cor[0] - 25, cor[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

plt.imshow(img_rgb)
plt.show()

plt.imshow(mask, cmap="gray")
plt.show()
