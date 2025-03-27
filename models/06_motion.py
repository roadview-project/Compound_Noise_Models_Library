import cv2
import numpy as np

def blur(img):
    # Based on https://stackoverflow.com/questions/7607464/implement-radial-blur-with-opencv
    w, h = img.shape[:2]
    # Centered based on ROADVIEW THI image
    center_x = (w/2) + 180
    center_y = (h/2) - 450
    blur = 0.002
    iterations = 7

    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)

    for i in range(iterations):
        tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)

    return img