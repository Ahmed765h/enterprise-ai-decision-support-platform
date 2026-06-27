import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_model(X_train, y_train, model_name=\"LogisticRegression\", params=None):
    """Trains a specified ML model and logs it with MLflow."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params if params else {})

        if model_name == \"LogisticRegression\":
            model = LogisticRegression(**(params if params else {}))
        elif model_name == \"RandomForestClassifier\":
            model = RandomForestClassifier(**(params if params else {}))
        elif model_name == \"XGBClassifier\":
            model = XGBClassifier(**(params if params else {}))
        elif model_name == \"LGBMClassifier\":
            model = LGBMClassifier(**(params if params else {}))
        else:
            raise ValueError(f\"Unknown model name: {model_name}\")

        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, \"model\")
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluates a trained model and logs metrics with MLflow."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=\"weighted\")
    recall = recall_score(y_test, y_pred, average=\"weighted\")
    f1 = f1_score(y_test, y_pred, average=\"weighted\")

    metrics = {
        \"accuracy\": accuracy,
        \"precision\": precision,
        \"recall\": recall,
        \"f1_score\": f1,
    }
    mlflow.log_metrics(metrics)
    return metrics

def save_model(model, path):
    """Saves a trained model to disk."""
    joblib.dump(model, path)

def load_model(path):
    """Loads a trained model from disk."""
    return joblib.load(path)
