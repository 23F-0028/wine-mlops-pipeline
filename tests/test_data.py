from src.data import load_data


def test_data_shape():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13


def test_no_missing_values():
    X_train, X_test, y_train, y_test = load_data()

    assert not (X_train != X_train).any()
    assert not (X_test != X_test).any()


def test_target_classes():
    X_train, X_test, y_train, y_test = load_data()

    assert set(y_train).issubset({0, 1, 2})
    assert set(y_test).issubset({0, 1, 2})
