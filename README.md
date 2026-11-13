# MLOps Assignment 1 – Wine Cultivar Classification Pipeline

## Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for Wine Cultivar Classification using the Scikit-learn Wine Dataset.

The project focuses on automating machine learning workflows, experiment tracking, model evaluation, model registration, reproducibility, testing, and continuous integration.

Two machine learning algorithms, Random Forest and Gradient Boosting, are trained and evaluated. MLflow is used to track experiments and manage registered models.

## Objectives

- Build a reproducible machine learning pipeline.
- Train and evaluate multiple classification models.
- Track experiments using MLflow.
- Log model parameters, metrics, and artifacts.
- Register the best-performing model in MLflow Model Registry.
- Apply automated model quality gates.
- Implement automated testing and code linting.
- Manage source code using Git and GitHub.
- Automate project workflows using GNU Make and GitHub Actions.

## Dataset Information

The project uses the built-in Wine Dataset provided by Scikit-learn.

| Attribute | Description |
|---|---|
| Dataset | Wine Dataset |
| Total Samples | 178 |
| Total Features | 13 |
| Target Classes | 3 |
| Problem Type | Multi-class Classification |
| Data Source | Scikit-learn |

The dataset contains numerical chemical measurements of wine samples belonging to three different cultivar classes.

## Technologies and Tools

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy and Pandas
- **Experiment Tracking:** MLflow
- **Model Registry:** MLflow Model Registry
- **Testing:** Pytest
- **Code Quality:** Flake8
- **Automation:** GNU Make
- **Version Control:** Git
- **Repository Hosting:** GitHub
- **CI/CD:** GitHub Actions

## Machine Learning Models

Two classification algorithms are implemented:

### 1. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions using majority voting.

### 2. Gradient Boosting Classifier

Gradient Boosting is an ensemble learning algorithm that builds models sequentially, where each new model attempts to correct errors made by previous models.

## Model Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Measures the percentage of correct predictions.
- **Macro F1-Score:** Calculates the average F1-score across all classes with equal importance.
- **Log Loss:** Measures the quality of predicted class probabilities.
- **Inference Latency:** Measures the time required to generate predictions.

## Experiment Tracking with MLflow

MLflow is used to track machine learning experiments.

The experiment name is:

`Wine-Cultivar-Classification`

MLflow records:

- Model parameters
- Evaluation metrics
- Training runs
- Model artifacts
- Experiment results

## Model Registry

The best-performing model is registered in the MLflow Model Registry.

| Attribute | Value |
|---|---|
| Registered Model | WineClassifier |
| Model Alias | champion |

The `champion` alias identifies the selected model in the model registry.

## Model Quality Gates

The project implements automated quality checks to ensure model reliability.

| Quality Requirement | Threshold |
|---|---|
| Macro F1-Score | >= 0.88 |
| Inference Latency | <= 30 ms |
| Valid Predicted Classes | 0, 1, 2 |

A model must satisfy the defined quality requirements to pass the validation process.

## Reproducibility

Reproducibility is maintained through:

- Fixed random state (`random_state=42`).
- Consistent dataset splitting.
- Recorded model parameters.
- MLflow experiment tracking.
- Version-controlled source code.
- Automated testing.

These practices help produce consistent and verifiable experimental results.

## Makefile Automation

GNU Make is used to simplify and automate common project operations.

| Command | Description |
|---|---|
| `make install` | Install project dependencies |
| `make lint` | Run code quality checks |
| `make test` | Execute automated tests |
| `make train` | Train and evaluate machine learning models |
| `make clean` | Remove generated files |

## Testing and Code Quality

Pytest is used to execute automated tests and verify the correctness of project components.

Flake8 is used for Python code linting and code quality checks.

These tools help maintain code reliability and consistency.

## Continuous Integration

GitHub Actions is used to automate project validation.

The CI workflow helps execute automated checks when code changes are pushed to the GitHub repository.

It supports:

- Dependency installation
- Code linting
- Automated testing

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- GNU Make
- pip

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd wine-mlops-pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
make install
```

### 4. Run Code Quality Checks

```bash
make lint
```

### 5. Run Automated Tests

```bash
make test
```

### 6. Train Machine Learning Models

```bash
make train
```

## Launch MLflow UI

To view experiment results and model information, execute:

```bash
mlflow ui
```

Open the following address in your browser:

http://127.0.0.1:5000

You can explore experiment runs, parameters, evaluation metrics, and registered models through the MLflow interface.

## Project Results

The experiments showed that Random Forest achieved the strongest overall performance among the evaluated models.

The reported Random Forest results included approximately:

- Accuracy: 0.9791
- Macro F1-Score: 0.9789
- Log Loss: 0.1630–0.1665

Gradient Boosting also achieved good classification performance, with reported Macro F1-scores ranging from approximately 0.9069 to 0.9188.

The selected model is managed through the MLflow Model Registry using the `WineClassifier` model name and `champion` alias.

## Conclusion

This project demonstrates the implementation of an end-to-end MLOps workflow for a multi-class wine classification problem.

It integrates machine learning model development, experiment tracking, model evaluation, quality gates, model registration, automated testing, code quality checks, version control, and continuous integration.

The implementation demonstrates how MLOps practices improve reproducibility, automation, maintainability, and reliability in machine learning projects.

---

**Project:** MLOps Assignment 1 – Wine Cultivar Classification Pipeline  
**Repository:** GitHub  
**Framework:** Scikit-learn and MLflow
