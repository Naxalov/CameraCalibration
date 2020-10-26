import cv2
import numpy as np
import os

#  320 * 2, 240 * 2

dist = np.array([[-0.02357899, 0.32398194, -0.01049933, 0.00349942, -0.2899601]], dtype=np.float32)
mtx = np.array([[551.35925517, 0., 248.33988363],
                [0., 551.84937767, 309.15983624],
                [0., 0., 1.]], dtype=np.float32)
# Photo
# DIR_INPUT = 'samples/photo'
# Squear
DIR_INPUT = '../samples/square'
PATH = os.path.join(os.getcwd(), DIR_INPUT)
images = os.listdir(PATH)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def show(img):
    # image = cv2.circle(img, (10,10), 2, (255,0,0), 2)
    # c = tuple(corners[0][0].astype(int))
    cv2.imshow('img', img)
    cv2.waitKey(0)    # img = draw(img, [100,100], imgpts)

    cv2.destroyAllWindows()
    print(1)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# W, H = 320 * 2, 240 * 2

img = cv2.imread(os.path.join(PATH, images[0]))
# get shape of image
W, H, _ = img.shape

model_points = np.array([
    [0.0, 0.0, 0.0],  # Nose tip
    [1.0, 0.0, 0.0],  # Nose tip
    [-1.0, 0.0, 0.0],  # Nose tip
    [-1.0, -1.0, 0.0],  # Nose tip

], np.float32)

image_points = np.array([
    [288, 323],  # Nose tip
    [312, 324],  # Nose tip
    [290, 346],  # Nose tip
    [312, 347],  # Nose tip

], dtype=np.float32)

for fname in images:

    img = cv2.imread(os.path.join(PATH, fname))
    img = cv2.resize(img, (H // 4, W // 4))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    # Find the rotation and translation vectors.
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, mtx, dist)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)

    img = draw(img, image_points, imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 0xff
    if k == 's':
        cv2.imwrite(fname[:6] + '.png', img)

cv2.destroyAllWindows()
print(1)
