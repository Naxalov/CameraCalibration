import os
import cv2
# Read image from folder

PATH = os.path.join(os.getcwd(), 'samples')
images = os.listdir(PATH)
img_path = os.path.join(images[0],PATH)
# Loads an image
img = cv2.imread(img_path)

# Check if image is loaded fine
if img is None:
    print('Error opening image!')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)