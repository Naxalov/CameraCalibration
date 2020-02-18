import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# input image in order calibration
INPUT_DIR = 'solepads/square'
PATH = os.path.join(os.getcwd(), INPUT_DIR)

images = os.listdir(PATH)
img = cv2.imread(os.path.join(PATH, images[0]))
# get shape of image
W, H, _ = img.shape

# RGB [13,15,12]
minBGR = np.array([4, 4, 4])
maxBGR = np.array([45, 45, 45])

mask = cv2.inRange(img, minBGR, maxBGR)
print(mask)
# result = cv2.bitwise_and(nemo, nemo, mask=mask)
plt.imshow(img)
plt.show()

plt.imshow(mask, cmap="gray")
plt.show()
