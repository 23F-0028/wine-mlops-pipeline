from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    """Load and validate the Wine dataset and create a stratified split."""

    wine = load_wine()

    X = wine.data
    y = wine.target

    # Validate feature count
    if X.shape[1] != 13:
        raise ValueError("Wine dataset must contain exactly 13 features.")

    # Validate missing values
    if np.isnan(X).any():
        raise ValueError("Dataset contains missing values.")

    # Create stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
