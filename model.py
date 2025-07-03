from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from joblib import dump, load
from tabulate import tabulate


def train_svm(X_train, y_train, kernel='linear'):
    """Train SVM with default parameters for each kernel type"""
    # Calculate class weights
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Default parameters for each kernel
    default_params = {
        'linear': {'C': 1.0},
        'poly': {'C': 1.0, 'degree': 3, 'gamma': 'scale'},
        'rbf': {'C': 1.0, 'gamma': 'scale'}
    }

    svm = SVC(
        kernel=kernel,
        class_weight=class_weights,
        probability=True,
        random_state=42,
        **default_params[kernel]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm.fit(X_train, y_train)

    return svm


def print_learning_curve_data(model, X, y, kernel_name):
    """Print learning curve data to terminal"""
    min_class_count = np.min(np.bincount(y))
    cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),  # Reduced points for cleaner output
        scoring='accuracy'
    )

    print(f"\nLearning Curve Data for {kernel_name} Kernel:")
    print("-" * 80)
    print(f"{'Training Examples':<20}{'Mean Train Accuracy':<20}{'Mean CV Accuracy':<20}")
    print("-" * 80)
    for size, train, test in zip(train_sizes,
                                 np.mean(train_scores, axis=1),
                                 np.mean(test_scores, axis=1)):
        print(f"{int(size):<20}{train:<20.4f}{test:<20.4f}")


def print_confusion_matrices(y_true, y_pred, class_names, kernel_name):
    """Print confusion matrices to terminal"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(f"\nConfusion Matrix for {kernel_name} Kernel (Raw Counts):")
    print("-" * 80)
    print(tabulate(cm, headers=class_names, showindex=class_names, tablefmt='grid'))

    print(f"\nNormalized Confusion Matrix for {kernel_name} Kernel:")
    print("-" * 80)
    print(tabulate(np.round(cm_norm, 4), headers=class_names, showindex=class_names, tablefmt='grid'))


def print_classification_report(y_true, y_pred, class_names, kernel_name):
    """Print classification report to terminal"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print(f"\nClassification Report for {kernel_name} Kernel:")
    print("-" * 80)
    print(tabulate(pd.DataFrame(report).transpose().round(4), headers='keys', tablefmt='grid'))


def print_roc_metrics(y_true, y_prob, class_names, kernel_name):
    """Print ROC/AUC metrics to terminal"""
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    roc_data = []

    print(f"\nROC/AUC Metrics for {kernel_name} Kernel:")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data.append([class_name, f"{roc_auc:.4f}"])

    print(tabulate(roc_data, headers=['Class', 'AUC Score'], tablefmt='grid'))


def print_kernel_results(X_train, y_train, y_true, y_pred, y_prob, class_names, kernel_name, train_acc=None):
    """Print all results for a single kernel"""
    print("\n" + "=" * 80)
    print(f"COMPLETE RESULTS FOR {kernel_name.upper()} KERNEL".center(80))
    print("=" * 80)

    # Print accuracy
    test_acc = accuracy_score(y_true, y_pred)
    print("\nACCURACY SCORES:")
    print("-" * 80)
    if train_acc is not None:
        print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    # Print learning curve data
    print_learning_curve_data(train_svm(X_train, y_train, kernel_name), X_train, y_train, kernel_name)

    # Print confusion matrices
    print_confusion_matrices(y_true, y_pred, class_names, kernel_name)

    # Print classification report
    print_classification_report(y_true, y_pred, class_names, kernel_name)

    # Print ROC metrics
    print_roc_metrics(y_true, y_prob, class_names, kernel_name)

    print("\n" + "=" * 80 + "\n")


def print_combined_results(results_dict, class_names):
    """Print combined comparison of all kernels"""
    print("\n" + "=" * 80)
    print("COMBINED RESULTS COMPARISON".center(80))
    print("=" * 80)

    # Accuracy comparison
    print("\nACCURACY COMPARISON:")
    print("-" * 80)
    acc_table = []
    for kernel, results in results_dict.items():
        acc_table.append([
            kernel,
            f"{results.get('train_acc', np.nan):.4f}",
            f"{results['test_acc']:.4f}",
            f"{results['test_acc'] - results.get('train_acc', results['test_acc']):.4f}"
        ])
    print(tabulate(acc_table,
                   headers=['Kernel', 'Train Acc', 'Test Acc', 'Difference'],
                   tablefmt='grid'))

    # ROC AUC comparison
    print("\nROC AUC COMPARISON:")
    print("-" * 80)
    roc_table = []
    for kernel, results in results_dict.items():
        row = [kernel]
        y_true_bin = label_binarize(results['y_true'], classes=np.arange(len(class_names)))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], results['y_prob'][:, i])
            row.append(f"{auc(fpr, tpr):.4f}")
        roc_table.append(row)
    print(tabulate(roc_table,
                   headers=['Kernel'] + class_names,
                   tablefmt='grid'))

    print("\n" + "=" * 80 + "\n")


# Visualization functions (unchanged from original)
def plot_combined_learning_curves(models, X, y, kernel_names):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red']

    for model, kernel, color in zip(models, kernel_names, colors):
        min_class_count = np.min(np.bincount(y))
        cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o--', color=color,
                 label=f'{kernel} (Train)')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color=color,
                 label=f'{kernel} (CV)')

    plt.title('Learning Curves Comparison', fontsize=14)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.5, 1.05)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_learning_curves.png', dpi=300)
    plt.show()


def plot_combined_confusion_matrices(y_true, preds_dict, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (kernel, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=False)
        ax.set_title(f'{kernel} Kernel', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)

    plt.suptitle('Normalized Confusion Matrices Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('combined_confusion_matrices.png', dpi=300)
    plt.show()


def plot_combined_roc_curves(y_true, probs_dict, class_names):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple']
    linestyles = ['-', '--', ':']

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    for i, class_name in enumerate(class_names):
        for (kernel, y_prob), linestyle in zip(probs_dict.items(), linestyles):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linestyle=linestyle, color=colors[i],
                     label=f'{kernel} {class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_roc_curves.png', dpi=300)
    plt.show()


def plot_combined_classification_reports(y_true, preds_dict, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (kernel, y_pred) in zip(axes, preds_dict.items()):
        report = classification_report(y_true, y_pred,
                                       target_names=class_names,
                                       output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :-1]

        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f',
                    ax=ax, cbar=False)
        ax.set_title(f'{kernel} Kernel\nAccuracy: {accuracy_score(y_true, y_pred):.2f}',
                     fontsize=12)

    plt.suptitle('Classification Reports Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('combined_classification_reports.png', dpi=300)
    plt.show()


def plot_accuracy_comparison(train_acc_dict, test_acc_dict):
    kernels = list(train_acc_dict.keys())
    train_acc = list(train_acc_dict.values())
    test_acc = list(test_acc_dict.values())

    x = np.arange(len(kernels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, train_acc, width, label='Training Accuracy', color='skyblue')
    bars2 = plt.bar(x + width / 2, test_acc, width, label='Testing Accuracy', color='orange')

    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training vs Testing Accuracy by Kernel', fontsize=14)
    plt.xticks(x, kernels)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    plt.show()


def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    return load(filename)