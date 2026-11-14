# MLOps AI Agent Architecture

## 1. System Overview
The **Intelligent Multi-Tool MLOps Agent** (`agent/orchestrator.py`) acts as a governed control plane across four core MLOps subsystems:
1. **DVC (`agent/tools/dvc_tools.py`)**: Dataset versioning, MD5 lineage tracking, status checks, and reproduction instructions.
2. **MLflow (`agent/tools/mlflow_tools.py`)**: Experiment comparison, metric evaluation (`f1_score`, `accuracy`), and Model Registry inspection (`iris-prod-model`).
3. **Apache Airflow 3 (`agent/tools/airflow_tools.py`)**: DAG discovery, workflow run auditing (`mlops_training_pipeline`), failed task diagnosis, and operator-approved triggering.
4. **Kubernetes & FastAPI Serving (`agent/tools/kubernetes_tools.py`, `api/main.py`)**: Cluster deployment monitoring (`iris-model-deployment`), pod health diagnostics, bounded scaling (`1..5` replicas), rolling restarts, and live `/health` endpoint probing.

## 2. Multi-Tool Correlation Flow (Steps 9–14)
When asked to evaluate model readiness and retraining necessity:
1. Queries MLflow for recent experiments, candidate runs, metrics, and registered model versions.
2. Reads DVC dataset metadata and compares the tracked MD5 hash against MLflow run tags/parameters.
3. Inspects Kubernetes deployment replica readiness, pod health, and the live serving `/health` endpoint.
4. Queries Airflow for recent training DAG runs and any quality-gate failures.
5. Correlates evidence across all four platforms and outputs a structured report separating **Known Facts**, **Uncertainties**, and **Recommended Actions**.

## 3. Reliability, Safety & Governance Guardrails
- **No Automatic Retraining**: Retraining is never triggered automatically on an audit query; it is only recommended when supported by concrete evidence (failed quality gate, dataset version mismatch, explicit freshness SLA, or supplied model degradation).
- **Non-Hallucination Policy**: When any external service (MLflow, DVC, Airflow, Kubernetes) is unreachable or returns an error, the agent records an explicit uncertainty rather than fabricating metrics or run IDs.
- **Kubernetes Security Guardrails**: Blocks access to system namespaces (`kube-system`, `kube-public`, `kube-node-lease`), rejects unauthorized mutations (`delete`, `exec`), and enforces bounded replica scaling (`1` to `5`).
- **Secret-Safe Structured Logging**: Emits JSON logs containing `request_id`, `tool_name`, `status`, and `duration_ms` while redacting tokens, passwords, and API keys.