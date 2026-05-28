"""Model training script with MLflow tracking."""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import yaml
import os

def load_config():
    with open("mlflow/experiments/config.yaml") as f:
        import re
        content = re.sub(r'\$\{(\w+)\}', lambda m: os.getenv(m.group(1), m.group(0)), f.read())
        return yaml.safe_load(content)

def train():
    config = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config["experiment"]["name"])

    # Replace with your actual data loading
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        mlflow.log_params(params)
        mlflow.set_tags(config["experiment"]["tags"])

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        f1 = f1_score(y_test, model.predict(X_test))
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

        thresholds = config["model_registry"]["staging_threshold"]
        if acc >= thresholds["accuracy"] and f1 >= thresholds["f1_score"]:
            mlflow.sklearn.log_model(model, "model", registered_model_name="production-model")
            print(f"Model registered: accuracy={acc:.3f}, f1={f1:.3f}")
        else:
            print(f"Model below threshold: accuracy={acc:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    train()
