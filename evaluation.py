from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_classification_report(y_test, y_pred, kernel_name):
    report = classification_report(y_test, y_pred, target_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
                                    output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', linewidths=0.5)
    plt.title(f'Classification Report ({kernel_name} Kernel)')
    plt.show()

def plot_confusion_matrix(y_test, y_pred, kernel_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'],
                yticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix ({kernel_name} Kernel)')
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy, kernel_name):
    plt.figure(figsize=(6, 4))
    plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title(f'{kernel_name.capitalize()} Kernel Accuracy')
    plt.show()