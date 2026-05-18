import cv2
import numpy as np

from skimage.feature import local_binary_pattern


def extract_lbp_features(image):

    # LBP parameters
    radius = 1
    n_points = 8 * radius

    # Compute LBP
    lbp = local_binary_pattern(
        image,
        n_points,
        radius,
        method='uniform'
    )

    # Histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    # Normalize histogram
    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)

    return hist