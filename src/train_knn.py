import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


def train_knn(X_train, y_train, k=3):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n==============================")
    print("Model Evaluation")
    print("==============================")

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test)
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title("Confusion Matrix")

    # Save image instead of show
    plt.savefig("results/confusion_matrix.png")

    plt.close()

    print("\nDa luu confusion matrix:")
    print("results/confusion_matrix.png")

    return predictions


def save_model(model, filename):

    joblib.dump(model, filename)

    print(f"\nModel da luu: {filename}")


def load_model(filename):

    model = joblib.load(filename)

    print(f"\nDa tai model: {filename}")

    return model