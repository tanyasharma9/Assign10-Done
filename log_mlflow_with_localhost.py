import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate accuracy as a target loss metric
accuracy = accuracy_score(y_test, y_pred)

mlflow.set_experiment("/iris-experiment")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="basic_lr_iris_model",
    )

from mlflow import MlflowClient

# Initialize an MLflow Client
client = MlflowClient()

def assign_alias_to_stage(model_name, stage, alias):
    """
    Assign an alias to the latest version of a registered model within a specified stage.

    :param model_name: The name of the registered model.
    :param stage: The stage of the model version for which the alias is to be assigned. Can be
                "Production", "Staging", "Archived", or "None".
    :param alias: The alias to assign to the model version.
    :return: None
    """
    latest_mv = client.get_latest_versions(model_name, stages=[stage])[0]
    client.set_registered_model_alias(model_name, alias, latest_mv.version)