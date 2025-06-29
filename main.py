import numpy as np
from data_loader import load_dataset
from feature_extraction import extract_hog_features, extract_color_histogram
from model import train_svm, tune_hyperparameters, plot_learning_curve, save_model
from evaluation import plot_classification_report, plot_confusion_matrix, plot_roc_curves
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


def main():
    # Load dataset
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset('dataset')
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Feature extraction
    print("\nExtracting features...")
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # Optional: Combine with color histograms
    X_train_hist = extract_color_histogram(X_train)
    X_test_hist = extract_color_histogram(X_test)

    # Combine features
    X_train_features = np.hstack([X_train_hog, X_train_hist])
    X_test_features = np.hstack([X_test_hog, X_test_hist])

    # SVM kernels to evaluate
    kernels = ['linear', 'poly', 'rbf']
    best_models = {}

    for kernel in kernels:
        print(f"\n{'=' * 50}\nEvaluating {kernel} kernel\n{'=' * 50}")
        start_time = time.time()

        # Hyperparameter tuning
        print("\nTuning hyperparameters...")
        best_model = tune_hyperparameters(X_train_features, y_train, kernel)

        # Training final model
        print("\nTraining final model...")
        best_model.fit(X_train_features, y_train)

        # Predictions
        y_train_pred = best_model.predict(X_train_features)
        y_test_pred = best_model.predict(X_test_features)
        y_test_prob = best_model.predict_proba(X_test_features)

        # Evaluation
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"\nTraining Accuracy ({kernel}): {train_acc:.4f}")
        print(f"Testing Accuracy ({kernel}): {test_acc:.4f}")

        # Visualization
        plot_learning_curve(best_model, X_train_features, y_train, kernel)
        plot_classification_report(y_test, y_test_pred, class_names, kernel)
        plot_confusion_matrix(y_test, y_test_pred, class_names, kernel)
        plot_roc_curves(y_test, y_test_prob, class_names, kernel)

        # Save best model
        best_models[kernel] = {
            'model': best_model,
            'test_accuracy': test_acc,
            'time': time.time() - start_time
        }

    # Compare all models
    print("\nModel Comparison:")
    for kernel, results in best_models.items():
        print(f"{kernel.upper()} Kernel - Accuracy: {results['test_accuracy']:.4f}, Time: {results['time']:.2f}s")

    # Save the best performing model
    best_kernel = max(best_models, key=lambda x: best_models[x]['test_accuracy'])
    save_model(best_models[best_kernel]['model'], f'best_svm_{best_kernel}.joblib')
    print(f"\nBest model ({best_kernel} kernel) saved to best_svm_{best_kernel}.joblib")


if __name__ == "__main__":
    main()