from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_classification_report(y_true, y_pred, classes, kernel_name):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f',
                linewidths=0.5, cbar=False)
    plt.title(f'Classification Report ({kernel_name} Kernel)\nAccuracy: {accuracy_score(y_true, y_pred):.2f}')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, kernel_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='lightgray')

    plt.title(f'Normalized Confusion Matrix ({kernel_name} Kernel)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true, y_prob, classes, kernel_name):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple']

    for i, color in zip(range(len(classes)), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC {classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({kernel_name} Kernel)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()