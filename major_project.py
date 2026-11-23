"""
Handwritten Digit Recognition using a Neural Network (MLP)
===========================================================
Major Project - Artificial Intelligence Training Batch

Loads the scikit-learn "digits" dataset (1,797 8x8 grayscale images of
handwritten digits 0-9), trains a fully-connected Neural Network
(Multi-Layer Perceptron) to classify them, evaluates its performance on
both the training and test sets, and visualizes the dataset, the
training process, sample predictions, and the confusion matrix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42


def load_data():
    """Load the digits dataset: 1797 samples, 8x8 images, 10 classes (0-9)."""
    digits = load_digits()
    return digits.data, digits.target, digits.images


def split_and_scale(X, y):
    """Split into train/test sets (80/20, stratified) and standardize features."""
    indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, idx_test, scaler


def build_and_train(X_train, y_train):
    """Define and train a fully-connected neural network (64 -> 64 -> 32 -> 10)."""

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=400,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    """Compute train accuracy (sanity check) and full test-set evaluation."""

    train_acc = accuracy_score(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, train_acc, test_acc, report, cm


def predict_single(model, scaler, X, y, index):
    """Run inference on one chosen sample - a simple interactive-style demo."""

    pixels = scaler.transform(X[index].reshape(1, -1))
    pred = model.predict(pixels)[0]

    print(
        f" Sample #{index}: True label = {y[index]}, "
        f"Predicted label = {pred}"
    )

    return pred


def plot_class_distribution(y, filename):
    """Bar chart showing how many training images exist per digit (0-9)."""

    counts = [np.sum(y == d) for d in range(10)]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(10), counts, color="#3b5bdb")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset Overview - Images per Digit Class")

    for i, c in enumerate(counts):
        ax.text(i, c + 1, str(c), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_loss_curve(model, filename):
    """Plot the training loss recorded by the network at every iteration."""

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(model.loss_curve_, color="#e8590c")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("Neural Network Training Loss Curve")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_sample_predictions(images, idx_test, y_test, y_pred, filename, n=16):
    """Save a grid of test images with True(T)/Predicted(P) labels."""

    fig, axes = plt.subplots(2, n // 2, figsize=(12, 3.4))

    for ax, img_idx, true_label, pred_label in zip(
        axes.ravel(), idx_test[:n], y_test[:n], y_pred[:n]
    ):
        ax.imshow(images[img_idx], cmap="gray_r")

        color = "green" if true_label == pred_label else "red"

        ax.set_title(
            f"T:{true_label} P:{pred_label}",
            color=color,
            fontsize=9
        )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, filename):
    """Save a heatmap of the confusion matrix."""

    fig, ax = plt.subplots(figsize=(5.5, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10)
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Digit Recognition")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


if __name__ == "__main__":

    X, y, images = load_data()

    print(f"Dataset shape: {X.shape}, Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test, idx_test, scaler = split_and_scale(X, y)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = build_and_train(X_train, y_train)

    y_pred, train_acc, test_acc, report, cm = evaluate(
        model, X_train, y_train, X_test, y_test
    )

    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:     {test_acc * 100:.2f}%")

    print("\nClassification Report (Test Set):\n", report)
    print("Confusion Matrix:\n", cm)

    print("\nInteractive-style demo - predicting a few individual samples:")

    for idx in [5, 42, 250, 900]:
        predict_single(model, scaler, X, y, idx)

    plot_class_distribution(y, "class_distribution.png")
    plot_loss_curve(model, "loss_curve.png")

    plot_sample_predictions(
        images, idx_test, y_test, y_pred, "sample_predictions.png"
    )

    plot_confusion_matrix(cm, "confusion_matrix.png")

    print(
        "\nSaved: class_distribution.png, loss_curve.png, "
        "sample_predictions.png, confusion_matrix.png"
    )