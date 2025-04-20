import os
import cv2
import numpy as np

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (100, 100))
            images.append(img_resized)
            labels.append(label)
    return images, labels

def load_dataset(root_folder):
    categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
    X, y = [], []
    for idx, category in enumerate(categories):
        train_folder = os.path.join(root_folder, 'training', category)
        test_folder = os.path.join(root_folder, 'testing', category)
        train_images, train_labels = load_images_from_folder(train_folder, idx)
        test_images, test_labels = load_images_from_folder(test_folder, idx)
        X.extend(train_images + test_images)
        y.extend(train_labels + test_labels)
    return np.array(X), np.array(y)