from skimage.feature import hog
import numpy as np
from tqdm import tqdm

def extract_hog_features(images):
    features = []
    for image in tqdm(images, desc="Extracting HOG features"):
        fd = hog(image,
                 orientations=9,
                 pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2),
                 visualize=False,
                 channel_axis=-1,
                 feature_vector=True)
        features.append(fd)
    return np.array(features)

def extract_color_histogram(images, bins=32):
    hist_features = []
    for image in tqdm(images, desc="Extracting color histograms"):
        # Extract histograms from each channel
        hist_r = np.histogram(image[:,:,0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(image[:,:,1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(image[:,:,2], bins=bins, range=(0, 256))[0]
        # Concatenate and normalize
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype("float")
        hist /= (hist.sum() + 1e-7)  # L1 normalization
        hist_features.append(hist)
    return np.array(hist_features)