import numpy as np
from sklearn.metrics import accuracy_score

from data_loader import load_dataset
from feature_extraction import extract_hog_features, extract_color_histogram
from model import (train_svm, plot_combined_learning_curves,
                   plot_combined_confusion_matrices,
                   plot_combined_roc_curves,
                   plot_combined_classification_reports,
                   save_model, plot_accuracy_comparison)
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

    # Combine with color histograms
    X_train_hist = extract_color_histogram(X_train)
    X_test_hist = extract_color_histogram(X_test)

    # Combine features
    X_train_features = np.hstack([X_train_hog, X_train_hist])
    X_test_features = np.hstack([X_test_hog, X_test_hist])

    # SVM kernels to evaluate
    kernels = ['linear', 'poly', 'rbf']
    models = {}
    preds_dict = {}
    probs_dict = {}
    train_accuracies = {}
    test_accuracies = {}

    # Train and evaluate all models
    for kernel in kernels:
        print(f"\n{'=' * 50}\nEvaluating {kernel} kernel\n{'=' * 50}")
        start_time = time.time()

        # Train model
        print("\nTraining model...")
        models[kernel] = train_svm(X_train_features, y_train, kernel)

        # Make predictions
        y_train_pred = models[kernel].predict(X_train_features)
        y_test_pred = models[kernel].predict(X_test_features)
        y_test_prob = models[kernel].predict_proba(X_test_features)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Store results
        preds_dict[kernel] = y_test_pred
        probs_dict[kernel] = y_test_prob
        train_accuracies[kernel] = train_acc
        test_accuracies[kernel] = test_acc

        print(f"\nTraining Accuracy ({kernel}): {train_acc:.4f}")
        print(f"Testing Accuracy ({kernel}): {test_acc:.4f}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Generate all visualizations
    print("\nGenerating visualizations...")

    # 1. Accuracy comparison bar chart
    plot_accuracy_comparison(train_accuracies, test_accuracies)

    # 2. Learning curves comparison
    plot_combined_learning_curves(
        [models[k] for k in kernels],
        X_train_features, y_train,
        kernels
    )

    # 3. Confusion matrices comparison
    plot_combined_confusion_matrices(
        y_test, preds_dict, class_names
    )

    # 4. ROC curves comparison
    plot_combined_roc_curves(
        y_test, probs_dict, class_names
    )

    # 5. Classification reports comparison
    plot_combined_classification_reports(
        y_test, preds_dict, class_names
    )

    # Save the best performing model based on test accuracy
    best_kernel = max(test_accuracies, key=test_accuracies.get)
    save_model(models[best_kernel], f'best_svm_{best_kernel}.joblib')
    print(f"\nBest model ({best_kernel} kernel) saved to best_svm_{best_kernel}.joblib")


if __name__ == "__main__":
    main()