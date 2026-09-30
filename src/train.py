import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data import load_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
RANDOM_STATE = 42


def get_models():
    """Return all model configurations for experimentation."""

    models = {
        "random_forest_1": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        "random_forest_2": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        "random_forest_3": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting_1": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting_2": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.10,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting_3": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.10,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
    }

    return models


def main():
    """Run experiments and register the best model."""

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "log_loss": "neg_log_loss"
    }

    models = get_models()

    best_run_id = None
    best_score = -1.0
    best_log_loss = float("inf")
    best_model_name = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:

            results = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )

            validation_accuracy = results["test_accuracy"].mean()
            validation_macro_f1 = results["test_macro_f1"].mean()
            validation_log_loss = -results["test_log_loss"].mean()

            train_accuracy = results["train_accuracy"].mean()
            train_macro_f1 = results["train_macro_f1"].mean()
            train_log_loss = -results["train_log_loss"].mean()

            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_family", type(model).__name__)

            mlflow.log_params(model.get_params())

            mlflow.log_metric(
                "validation_accuracy",
                validation_accuracy
            )
            mlflow.log_metric(
                "validation_macro_f1",
                validation_macro_f1
            )
            mlflow.log_metric(
                "validation_log_loss",
                validation_log_loss
            )

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_macro_f1", train_macro_f1)
            mlflow.log_metric("train_log_loss", train_log_loss)

            # Train the candidate on the complete training split
            model.fit(X_train, y_train)

            predictions = model.predict(X_train)

            signature = infer_signature(
                X_train,
                predictions
            )

            input_example = X_train[:5]

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Select best model using validation Macro F1.
            # If tied, use lower validation Log Loss.
            if (
                validation_macro_f1 > best_score
                or (
                    validation_macro_f1 == best_score
                    and validation_log_loss < best_log_loss
                )
            ):
                best_score = validation_macro_f1
                best_log_loss = validation_log_loss
                best_run_id = run.info.run_id
                best_model_name = model_name

            print(f"\nModel: {model_name}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation Macro F1: {validation_macro_f1:.4f}")
            print(f"Validation Log Loss: {validation_log_loss:.4f}")

    # Register the best model
    client = MlflowClient()

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version
    )

    print("\nBest Model:", best_model_name)
    print("Best Validation Macro F1:", f"{best_score:.4f}")
    print("Registered Model:", REGISTERED_MODEL_NAME)
    print("Champion Version:", registered_model.version)


if __name__ == "__main__":
    main()
