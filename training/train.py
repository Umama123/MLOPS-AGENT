import os
import time
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_dvc_md5(dvc_file: str = "data/dataset.csv.dvc") -> str:
    try:
        with open(dvc_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data["outs"][0]["md5"]
    except Exception:
        return "unknown"


def train_and_log(
    experiment_name: str,
    model_type: str,
    params: dict,
    register_as: str | None = None,
):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv("data/dataset.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    dataset_md5 = get_dvc_md5()

    with mlflow.start_run(run_name=f"{model_type}-run") as run:
        start_t = time.perf_counter()

        if model_type == "random_forest":
            clf = RandomForestClassifier(**params, random_state=42)
        elif model_type == "gradient_boosting":
            clf = GradientBoostingClassifier(**params, random_state=42)
        else:
            clf = LogisticRegression(**params, random_state=42)

        clf.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - start_t) * 1000

        inf_start = time.perf_counter()
        y_pred_test = clf.predict(X_test)
        inference_latency_ms = ((time.perf_counter() - inf_start) / len(X_test)) * 1000
        y_pred_train = clf.predict(X_train)

        train_f1 = f1_score(y_train, y_pred_train, average="weighted")
        test_f1 = f1_score(y_test, y_pred_test, average="weighted")
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average="weighted")
        rec = recall_score(y_test, y_pred_test, average="weighted")
        overfit_gap = round(train_f1 - test_f1, 4)

        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.set_tag("dvc_dataset_md5", dataset_md5)
        mlflow.set_tag("dataset_path", "data/dataset.csv")

        mlflow.log_metric("f1_score", round(test_f1, 4))
        mlflow.log_metric("train_f1_score", round(train_f1, 4))
        mlflow.log_metric("overfit_gap", overfit_gap)
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("train_time_ms", round(train_time_ms, 2))
        mlflow.log_metric("inference_latency_ms", round(inference_latency_ms, 4))

        # Use cloudpickle serialization to avoid skops compatibility issues
        if register_as:
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model",
                registered_model_name=register_as,
                serialization_format="cloudpickle",
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model",
                serialization_format="cloudpickle",
            )

        print(
            f"Run {run.info.run_id[:8]} | {model_type} | test_f1={test_f1:.4f} | overfit_gap={overfit_gap:.4f}"
        )


if __name__ == "__main__":
    train_and_log("iris-classifier-v1", "logistic_regression", {"C": 0.1, "max_iter": 200})
    train_and_log("iris-classifier-v1", "logistic_regression", {"C": 1.0, "max_iter": 200})
    train_and_log("iris-classifier-v2", "random_forest", {"n_estimators": 20, "max_depth": 2})
    train_and_log("iris-classifier-v2", "random_forest", {"n_estimators": 100, "max_depth": 5}, register_as="iris-prod-model")
    train_and_log("iris-classifier-v2", "gradient_boosting", {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}, register_as="iris-prod-model")
    train_and_log("iris-classifier-v2", "random_forest", {"n_estimators": 300, "max_depth": 15})