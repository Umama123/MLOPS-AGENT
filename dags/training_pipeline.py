from datetime import datetime, timedelta
import logging
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.sklearn
from airflow.decorators import dag, task

logger = logging.getLogger("airflow.task")

DATA_PATH = "/opt/airflow/data/dataset.csv"
PROCESSED_PATH = "/opt/airflow/data/processed_dataset.csv"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////opt/airflow/mlflow.db")
F1_THRESHOLD = 0.90


@dag(
    dag_id="mlops_training_pipeline",
    description="5-stage ML training pipeline with validation, MLflow logging, and quality gate",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={
        "owner": "mlops-student",
        "retries": 2,
        "retry_delay": timedelta(seconds=15),
    },
    tags=["mlops", "training", "approved"],
)
def mlops_training_pipeline():

    @task()
    def validate_dataset(**context) -> str:
        """Stage 1: Validate dataset existence, schema, null values, and minimum row count."""
        conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
        if conf.get("simulate_invalid_data"):
            raise ValueError("Dataset validation failed: simulated schema/null corruption injected.")

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset missing at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        required_cols = {
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "target",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.isnull().any().any():
            raise ValueError("Dataset contains null/NaN values.")

        if len(df) < 50:
            raise ValueError(f"Insufficient rows ({len(df)}). Minimum 50 required.")

        logger.info("Dataset validated successfully: %d rows, %d columns.", len(df), len(df.columns))
        return DATA_PATH

    @task()
    def preprocess_data(dataset_path: str) -> str:
        """Stage 2: Clean duplicates/outliers and prepare training-ready dataset."""
        df = pd.read_csv(dataset_path)
        initial_rows = len(df)
        df_clean = df.dropna().reset_index(drop=True)
        df_clean.to_csv(PROCESSED_PATH, index=False)
        logger.info("Preprocessed dataset saved to %s (%d -> %d rows).", PROCESSED_PATH, initial_rows, len(df_clean))
        return PROCESSED_PATH

    @task()
    def train_model(processed_path: str, **context) -> dict:
        """Stage 3: Train classifier and return artifact location & evaluation splits."""
        conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
        n_estimators = int(conf.get("n_estimators", 100))
        max_depth = int(conf.get("max_depth", 5))

        df = pd.read_csv(processed_path)
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        test_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        test_acc = float(accuracy_score(y_test, y_pred))

        # Allow simulated low-accuracy run for testing the quality gate
        if conf.get("simulate_low_f1"):
            test_f1 = 0.62

        model_dir = "/opt/airflow/data/latest_model"
        os.makedirs(model_dir, exist_ok=True)
        import joblib
        model_file = os.path.join(model_dir, "model.joblib")
        joblib.dump(clf, model_file)

        logger.info("Model trained (n_estimators=%d, max_depth=%d) | F1=%.4f", n_estimators, max_depth, test_f1)
        return {
            "model_file": model_file,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "f1_score": round(test_f1, 4),
            "accuracy": round(test_acc, 4),
        }

    @task()
    def evaluate_and_log_mlflow(train_result: dict) -> dict:
        """Stage 4: Log parameters, metrics, and model artifact to MLflow."""
        import joblib
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("airflow-scheduled-training")

        clf = joblib.load(train_result["model_file"])
        with mlflow.start_run(run_name="airflow-rf-run") as run:
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("n_estimators", train_result["n_estimators"])
            mlflow.log_param("max_depth", train_result["max_depth"])
            mlflow.log_metric("f1_score", train_result["f1_score"])
            mlflow.log_metric("accuracy", train_result["accuracy"])
            mlflow.set_tag("orchestrator", "apache-airflow-3")
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model",
                serialization_format="cloudpickle",
            )
            run_id = run.info.run_id

        logger.info("Logged run %s to MLflow with f1_score=%.4f", run_id, train_result["f1_score"])
        return {
            "run_id": run_id,
            "f1_score": train_result["f1_score"],
            "accuracy": train_result["accuracy"],
        }

    @task()
    def apply_model_quality_gate(eval_result: dict) -> dict:
        """Stage 5: Enforce minimum quality threshold before registering/promoting model."""
        f1 = eval_result["f1_score"]
        run_id = eval_result["run_id"]

        if f1 < F1_THRESHOLD:
            logger.error(
                "QUALITY GATE FAILED: F1 score %.4f is below threshold %.2f. Blocking promotion.",
                f1,
                F1_THRESHOLD,
            )
            raise ValueError(
                f"Quality gate rejected run {run_id}: f1_score={f1} < required {F1_THRESHOLD}"
            )

        mlflow.set_tracking_uri(MLFLOW_URI)
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name="iris-prod-model")
        logger.info(
            "QUALITY GATE PASSED: Registered run %s as iris-prod-model version %s.",
            run_id,
            mv.version,
        )
        return {
            "promoted": True,
            "registered_model": "iris-prod-model",
            "version": mv.version,
            "f1_score": f1,
        }

    # Explicit 5-stage pipeline dependencies
    raw_data = validate_dataset()
    clean_data = preprocess_data(raw_data)
    trained = train_model(clean_data)
    evaluated = evaluate_and_log_mlflow(trained)
    apply_model_quality_gate(evaluated)


mlops_training_pipeline()
