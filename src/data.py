from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    # Load the Wine dataset
    wine = load_wine()

    X = wine.data
    y = wine.target

    # Check that the dataset has exactly 13 features
    assert X.shape[1] == 13

    # Check that there are no missing values
    assert not (X != X).any()

    # Split data into 80% training and 20% testing
    # Stratify keeps the class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test