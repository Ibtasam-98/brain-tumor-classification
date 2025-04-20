import numpy as np
from skimage.feature import hog

def extract_hog_features(images):
    features = []
    for image in images:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        features.append(fd)
    return np.array(features)