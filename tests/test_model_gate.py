import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data import load_data


def get_test_model():
    """Create the deterministic model used by the quality gate."""

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )


def test_validation_macro_f1_gate():
    """Validation Macro F1 must be at least 0.88."""

    X_train, X_test, y_train, y_test = load_data()

    model = get_test_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    assert macro_f1 >= 0.88


def test_batch_inference_time_gate():
    """Batch inference time must be 30 ms or less."""

    X_train, X_test, y_train, y_test = load_data()

    model = get_test_model()
    model.fit(X_train, y_train)

    start_time = time.perf_counter()
    model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert elapsed_ms <= 30


def test_output_classes_gate():
    """Predictions must contain only classes 0, 1, and 2."""

    X_train, X_test, y_train, y_test = load_data()

    model = get_test_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(predictions).issubset({0, 1, 2})
