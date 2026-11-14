import json
import logging
from unittest.mock import patch
import pytest

from agent.orchestrator import MLOpsOrchestratorAgent


@pytest.fixture
def agent():
    return MLOpsOrchestratorAgent(serving_health_url="http://localhost:30080/health")


def _mock_healthy_stack():
    """Helper returning healthy mocked responses across MLflow, DVC, K8s, and Airflow."""
    return {
        "mlflow.list_experiments": {
            "status": "ok",
            "experiments": [{"name": "iris-classification", "experiment_id": "1"}],
        },
        "mlflow.compare_runs_and_recommend": {
            "status": "ok",
            "recommended_candidate": {
                "run_id": "run_abc123",
                "all_metrics": {"f1_score": 0.98, "accuracy": 0.98},
                "dvc_dataset_md5": "hash_v1",
            },
        },
        "mlflow.get_registered_models": {"status": "ok", "models": []},
        "mlflow.get_run_details": {
            "status": "ok",
            "run_id": "run_abc123",
            "metrics": {"f1_score": 0.98},
            "params": {"dataset_version": "hash_v1"},
            "tags": {},
        },
        "dvc.get_dataset_metadata": {"status": "ok", "md5": "hash_v1"},
        "dvc.get_dvc_status": {"status": "ok", "has_changes": False},
        "dvc.list_dataset_versions": {"status": "ok", "versions": ["v1"]},
        "k8s.get_cluster_status": {
            "status": "ok",
            "deployments": [
                {
                    "name": "iris-model-deployment",
                    "desired_replicas": 2,
                    "ready_replicas": 2,
                    "image": "iris-model-service:v1",
                }
            ],
            "pods": [],
        },
        "k8s.diagnose_unhealthy_pods": {
            "status": "ok",
            "healthy": True,
            "unhealthy_count": 0,
            "issues": [],
        },
        "airflow.list_dags": {
            "status": "ok",
            "dags": [{"dag_id": "mlops_training_pipeline", "is_paused": False}],
        },
        "airflow.get_dag_runs": {
            "status": "ok",
            "runs": [{"dag_run_id": "manual__01", "state": "success"}],
        },
        "airflow.diagnose_failed_tasks": {
            "status": "ok",
            "failed_tasks": [],
        },
    }


# 1. Successful multi-tool selection and execution
def test_successful_multi_tool_execution(agent):
    mocks = _mock_healthy_stack()
    with patch.dict(
        agent.tool_registry,
        {k: (lambda v=v: lambda **kw: v)() for k, v in mocks.items()},
    ), patch.object(
        agent,
        "_check_serving_endpoint",
        return_value={
            "status": "ok",
            "http_code": 200,
            "response": {"model_version": "v1"},
        },
    ):
        report = agent.execute_model_audit_and_retrain_decision(
            query="Find best recent model and check deployment status"
        )
        assert report.correlated_evidence["mlflow"]["best_run_id"] == "run_abc123"
        assert report.correlated_evidence["dvc"]["current_tracked_md5"] == "hash_v1"
        assert report.correlated_evidence["kubernetes"]["pods_healthy"] is True
        assert report.correlated_evidence["airflow"]["latest_run_state"] == "success"
        assert report.retraining_initiated is False
        assert len(report.tool_trace) >= 12


# 2. Mocked responses when services are unavailable
def test_mocked_unavailable_services_handled_gracefully(agent):
    unavailable = {
        k: (lambda **kw: {"status": "error", "error": "Connection refused"})
        for k in agent.tool_registry
    }
    with patch.dict(agent.tool_registry, unavailable), patch.object(
        agent,
        "_check_serving_endpoint",
        return_value={"status": "unreachable", "error": "Connection refused"},
    ):
        report = agent.execute_model_audit_and_retrain_decision(
            query="Check model and deployment status"
        )
        assert report.retraining_initiated is False
        assert len(report.uncertainties) >= 4
        assert any("MLflow service is unavailable" in u for u in report.uncertainties)
        assert any("Kubernetes cluster status is unavailable" in u for u in report.uncertainties)
        assert any("Airflow workflow status is unavailable" in u for u in report.uncertainties)


# 3. Proof the agent does NOT invent operational results when a tool is unavailable
def test_agent_does_not_invent_results_when_tool_unavailable(agent):
    mocks = _mock_healthy_stack()
    mocks["mlflow.list_experiments"] = {"status": "error", "error": "MLflow server down"}
    mocks["mlflow.compare_runs_and_recommend"] = {"status": "error", "error": "MLflow server down"}

    with patch.dict(
        agent.tool_registry,
        {k: (lambda v=v: lambda **kw: v)() for k, v in mocks.items()},
    ), patch.object(
        agent,
        "_check_serving_endpoint",
        return_value={"status": "ok", "http_code": 200, "response": {"model_version": "v1"}},
    ):
        report = agent.execute_model_audit_and_retrain_decision(query="Find best model")
        # Agent must NOT fabricate a run_id or metrics
        assert report.correlated_evidence["mlflow"]["best_run_id"] is None
        assert report.correlated_evidence["mlflow"]["metrics"] == {}
        assert not any("identified best run" in f for f in report.known_facts)
        assert any("MLflow service is unavailable" in u for u in report.uncertainties)


# 4. Tool timeout handling
def test_tool_timeout_handling(agent):
    def _raise_timeout(**kwargs):
        raise TimeoutError("Airflow REST API exceeded 10s timeout")

    with patch.dict(agent.tool_registry, {"airflow.get_dag_runs": _raise_timeout}):
        res = agent.invoke_tool("airflow.get_dag_runs", dag_id="mlops_training_pipeline")
        assert res["status"] == "timeout"
        assert "timed out" in res["error"]


# 5. Invalid requests & unknown tools handling
def test_invalid_request_and_unknown_tool(agent):
    unknown_res = agent.invoke_tool("nonexistent.tool", foo="bar")
    assert unknown_res["status"] == "error"
    assert "Unknown tool" in unknown_res["error"]

    # Extra invalid kwargs should be safely filtered rather than crashing valid tools
    with patch.dict(
        agent.tool_registry,
        {"dvc.get_dvc_status": lambda: {"status": "ok", "has_changes": False}},
    ):
        clean_res = agent.invoke_tool("dvc.get_dvc_status", invalid_arg=999)
        assert clean_res["status"] == "ok"


# 6. Service failures (HTTP 500 / API error)
def test_service_failure_propagation(agent):
    with patch.dict(
        agent.tool_registry,
        {
            "k8s.get_cluster_status": lambda **kw: {
                "status": "error",
                "error": "K8s API Server HTTP 503 Service Unavailable",
            }
        },
    ):
        res = agent.invoke_tool("k8s.get_cluster_status")
        assert res["status"] == "error"
        assert "503" in res["error"]


# 7. Unauthorized Kubernetes modifications are rejected
def test_unauthorized_kubernetes_modifications_rejected(agent):
    # Attempting to delete a deployment
    res_delete = agent.handle_query("delete kubernetes deployment iris-model-deployment")
    assert res_delete["status"] == "rejected"
    assert "Unauthorized Kubernetes modification" in res_delete["error"]

    # Attempting to touch protected system namespace kube-system
    res_sys = agent.invoke_tool(
        "k8s.scale_deployment",
        deployment_name="coredns",
        replicas=2,
        namespace="kube-system",
    )
    assert res_sys["status"] == "rejected"
    assert "kube-system" in res_sys["error"]

    # Attempting unsafe replica count > 5
    res_scale = agent.invoke_tool(
        "k8s.scale_deployment",
        deployment_name="iris-model-deployment",
        replicas=20,
        namespace="default",
    )
    assert res_scale["status"] == "rejected"
    assert "between 1 and 5" in res_scale["error"]


# 8. Structured logs record tool_name, status, duration_ms, request_id without secrets
def test_structured_logging_records_metadata_without_secrets(agent, caplog):
    with patch.dict(
        agent.tool_registry,
        {"airflow.list_dags": lambda **kw: {"status": "ok", "dags": []}},
    ), caplog.at_level(logging.INFO, logger="mlops_agent"):
        agent.invoke_tool(
            "airflow.list_dags",
            request_id="req-test-999",
            api_token="super_secret_token_12345",
            password="my_admin_password",
        )

    assert len(caplog.records) >= 1
    log_payload = json.loads(caplog.records[-1].getMessage())
    assert log_payload["request_id"] == "req-test-999"
    assert log_payload["tool_name"] == "airflow.list_dags"
    assert log_payload["status"] == "ok"
    assert isinstance(log_payload["duration_ms"], float)
    # Secrets must be redacted
    raw_log = caplog.records[-1].getMessage()
    assert "super_secret_token_12345" not in raw_log
    assert "my_admin_password" not in raw_log
    assert "***REDACTED***" in raw_log


# 9. Deliberately Injected Failure #1: Airflow Quality Gate Failure Diagnosis
def test_injected_failure_airflow_quality_gate(agent):
    mocks = _mock_healthy_stack()
    mocks["airflow.get_dag_runs"] = {
        "status": "ok",
        "runs": [{"dag_run_id": "manual__failed_01", "state": "failed"}],
    }
    mocks["airflow.diagnose_failed_tasks"] = {
        "status": "ok",
        "failed_tasks": [
            {
                "task_id": "quality_gate",
                "state": "failed",
                "reason": "Validation F1 score 0.71 below threshold 0.85",
            }
        ],
    }
    with patch.dict(
        agent.tool_registry,
        {k: (lambda v=v: lambda **kw: v)() for k, v in mocks.items()},
    ), patch.object(
        agent,
        "_check_serving_endpoint",
        return_value={"status": "ok", "http_code": 200, "response": {"model_version": "v1"}},
    ):
        report = agent.execute_model_audit_and_retrain_decision(
            query="Check pipeline and whether retraining is needed"
        )
        assert report.retraining_initiated is False
        assert any("quality_gate" in str(f) for f in report.known_facts)
        assert any("Retraining IS warranted" in a for a in report.recommended_actions)


# 10. Deliberately Injected Failure #2: Kubernetes CrashLoopBackOff Pod Diagnosis
def test_injected_failure_kubernetes_crashloopbackoff(agent):
    mocks = _mock_healthy_stack()
    mocks["k8s.get_cluster_status"] = {
        "status": "ok",
        "deployments": [
            {
                "name": "iris-model-deployment",
                "desired_replicas": 2,
                "ready_replicas": 0,
                "image": "iris-model-service:broken",
            }
        ],
        "pods": [{"name": "iris-model-deployment-xyz", "phase": "Running", "ready": False}],
    }
    mocks["k8s.diagnose_unhealthy_pods"] = {
        "status": "ok",
        "healthy": False,
        "unhealthy_count": 1,
        "issues": [
            {
                "pod_name": "iris-model-deployment-xyz",
                "reasons": ["Waiting: CrashLoopBackOff - Back-off restarting failed container"],
            }
        ],
    }
    with patch.dict(
        agent.tool_registry,
        {k: (lambda v=v: lambda **kw: v)() for k, v in mocks.items()},
    ), patch.object(
        agent,
        "_check_serving_endpoint",
        return_value={"status": "unreachable", "error": "Connection reset"},
    ):
        report = agent.execute_model_audit_and_retrain_decision(
            query="Diagnose deployment health"
        )
        assert report.correlated_evidence["kubernetes"]["pods_healthy"] is False
        assert any("CrashLoopBackOff" in f for f in report.known_facts)
        assert any("k8s.get_pod_logs" in a for a in report.recommended_actions)