import numpy as np
from sklearn.metrics import accuracy_score
from data_loader import load_dataset
from feature_extraction import extract_hog_features, extract_color_histogram
from model import (train_svm, plot_combined_learning_curves,
                  plot_combined_confusion_matrices,
                  plot_combined_roc_curves,
                  plot_combined_classification_reports,
                  save_model, plot_accuracy_comparison,
                  print_kernel_results, print_combined_results)
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
    results_dict = {}

    # Train and evaluate all models
    for kernel in kernels:
        print(f"\n{'=' * 50}\nEvaluating {kernel} kernel\n{'=' * 50}")
        start_time = time.time()

        # Train model
        print("\nTraining model...")
        model = train_svm(X_train_features, y_train, kernel)

        # Make predictions
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        y_test_prob = model.predict_proba(X_test_features)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Store results
        results_dict[kernel] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'y_pred': y_test_pred,
            'y_prob': y_test_prob,
            'y_true': y_test
        }

        # Print detailed results for this kernel
        print_kernel_results(X_train_features, y_train, y_test, y_test_pred,
                           y_test_prob, class_names, kernel, train_acc)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Print combined comparison of all kernels
    print_combined_results(results_dict, class_names)

    # Generate all visualizations
    print("\nGenerating visualizations...")

    # 1. Accuracy comparison bar chart
    plot_accuracy_comparison(
        {k: results_dict[k]['train_acc'] for k in kernels},
        {k: results_dict[k]['test_acc'] for k in kernels}
    )

    # 2. Learning curves comparison
    plot_combined_learning_curves(
        [results_dict[k]['model'] for k in kernels],
        X_train_features, y_train,
        kernels
    )

    # 3. Confusion matrices comparison
    plot_combined_confusion_matrices(
        y_test,
        {k: results_dict[k]['y_pred'] for k in kernels},
        class_names
    )

    # 4. ROC curves comparison
    plot_combined_roc_curves(
        y_test,
        {k: results_dict[k]['y_prob'] for k in kernels},
        class_names
    )

    # 5. Classification reports comparison
    plot_combined_classification_reports(
        y_test,
        {k: results_dict[k]['y_pred'] for k in kernels},
        class_names
    )

    # Save the best performing model
    best_kernel = max(results_dict, key=lambda x: results_dict[x]['test_acc'])
    save_model(results_dict[best_kernel]['model'], f'best_svm_{best_kernel}.joblib')
    print(f"\nBest model ({best_kernel} kernel) saved to best_svm_{best_kernel}.joblib")

if __name__ == "__main__":
    main()