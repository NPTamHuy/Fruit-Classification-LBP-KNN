import cv2
import numpy as np


def extract_color_histogram(image):

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Histogram
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    # Normalize
    hist = cv2.normalize(hist, hist).flatten()

    return hist