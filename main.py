from sklearn.model_selection import train_test_split
import numpy as np
from data_loader import load_dataset
from feature_extraction import extract_hog_features
from model import train_svm, predict
from evaluation import plot_classification_report, plot_confusion_matrix, plot_accuracy
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
root_folder = 'dataset'
images, labels = load_dataset(root_folder)

# Extract HOG features
features = extract_hog_features(images)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# SVM kernels and their parameters
kernels = ['linear', 'poly', 'rbf']
kernel_params = {
    'linear': {'C': 0.2},
    'poly': {'C': 1, 'degree': 3},
    'rbf': {'C': 1, 'gamma': 'scale'}
}

for kernel in kernels:
    print(f"\n--- SVM with {kernel} kernel ---")
    params = kernel_params[kernel]
    svm_model = train_svm(X_train, y_train, kernel, params)

    y_train_pred = predict(svm_model, X_train)
    y_test_pred = predict(svm_model, X_test)

    # Training and Testing Accuracy
    train_accuracy = np.mean(y_train == y_train_pred)
    test_accuracy = np.mean(y_test == y_test_pred)

    print(f'Training Accuracy ({kernel} kernel): {train_accuracy * 100:.2f}%')
    print(f'Testing Accuracy ({kernel} kernel): {test_accuracy * 100:.2f}%')

    # Evaluation
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report:\n",
          classification_report(y_test, y_test_pred, target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))

    # Plotting the metrics
    plot_classification_report(y_test, y_test_pred, kernel)
    plot_confusion_matrix(y_test, y_test_pred, kernel)
    plot_accuracy(train_accuracy, test_accuracy, kernel)