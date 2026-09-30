import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, log_loss

from data import load_data


MODEL_URI = "models:/WineClassifier@champion"


def main():
    """Evaluate the registered champion model on test data."""

    X_train, X_test, y_train, y_test = load_data()

    # Load champion model from MLflow Model Registry
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Generate predictions
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    # Calculate final test metrics
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )
    loss = log_loss(
        y_test,
        probabilities
    )

    print("Final Test Evaluation")
    print("---------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
