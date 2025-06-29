import os
import cv2
import numpy as np
from sklearn.utils import shuffle


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
    return np.array(images), np.array(labels)


def load_dataset(root_folder):
    categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
    X_train, y_train = [], []
    X_test, y_test = [], []

    for idx, category in enumerate(categories):
        train_folder = os.path.join(root_folder, 'training', category)
        test_folder = os.path.join(root_folder, 'testing', category)

        train_images, train_labels = load_images_from_folder(train_folder, idx)
        test_images, test_labels = load_images_from_folder(test_folder, idx)

        X_train.extend(train_images)
        y_train.extend(train_labels)
        X_test.extend(test_images)
        y_test.extend(test_labels)

    # Convert to numpy arrays and shuffle
    X_train, y_train = shuffle(np.array(X_train), np.array(y_train), random_state=42)
    X_test, y_test = shuffle(np.array(X_test), np.array(y_test), random_state=42)

    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    return X_train, y_train, X_test, y_test