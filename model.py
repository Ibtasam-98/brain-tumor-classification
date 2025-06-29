from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import warnings
from joblib import dump, load


def train_svm(X_train, y_train, kernel, params=None):
    if params is None:
        params = {}

    # Calculate class weights
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    svm = SVC(
        kernel=kernel,
        class_weight=class_weights,
        probability=True,
        random_state=42,
        **params
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm.fit(X_train, y_train)

    return svm


def tune_hyperparameters(X_train, y_train, kernel):
    param_grid = {
        'linear': {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': ['balanced', None]
        },
        'poly': {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'coef0': [0, 1, 10],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced', None]
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced', None]
        }
    }

    min_class_count = np.min(np.bincount(y_train))
    n_splits = min(5, min_class_count)

    grid_search = GridSearchCV(
        SVC(kernel=kernel, random_state=42, probability=True),  # Add probability=True here
        param_grid[kernel],
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        error_score='raise'
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    print(f"\nBest parameters for {kernel} kernel:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_



def plot_learning_curve(estimator, X, y, kernel_name):
    min_class_count = np.min(np.bincount(y))
    cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes,
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                     np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                     alpha=0.1, color="g")

    plt.title(f"Learning Curve ({kernel_name} Kernel)")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.05)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    return load(filename)