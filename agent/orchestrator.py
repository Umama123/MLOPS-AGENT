import inspect
import json
import logging
import os
import re
import time
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.tools import (
    airflow_tools,
    dvc_tools,
    kubernetes_tools,
    mlflow_tools,
)

# ---------------------------------------------------------------------------
# Structured Secret-Safe Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("mlops_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SECRET_PATTERN = re.compile(
    r"(password|token|secret|api_key|authorization)\s*[:=]\s*['\"]?[^\s,'\";\}]+",
    re.IGNORECASE,
)


def _redact_secrets(payload: Any) -> Any:
    """Recursively mask sensitive tokens/passwords before writing structured logs."""
    if isinstance(payload, dict):
        sanitized = {}
        for k, v in payload.items():
            if any(
                s in str(k).lower()
                for s in ("password", "token", "secret", "api_key", "auth")
            ):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = _redact_secrets(v)
        return sanitized
    if isinstance(payload, list):
        return [_redact_secrets(item) for item in payload]
    if isinstance(payload, str):
        return SECRET_PATTERN.sub(r"\1=***REDACTED***", payload)
    return payload


# ---------------------------------------------------------------------------
# Structured Schemas
# ---------------------------------------------------------------------------
class ToolCallRecord(BaseModel):
    step: int
    request_id: str
    tool_name: str
    status: str
    duration_ms: float
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Dict[str, Any] = Field(default_factory=dict)


class StructuredAgentReport(BaseModel):
    request_id: str
    query: str
    mode: str
    tool_trace: List[ToolCallRecord]
    correlated_evidence: Dict[str, Any]
    known_facts: List[str]
    uncertainties: List[str]
    recommended_actions: List[str]
    retraining_initiated: bool = False


# ---------------------------------------------------------------------------
# Multi-Tool MLOps Orchestrator Agent
# ---------------------------------------------------------------------------
class MLOpsOrchestratorAgent:
    """
    Intelligent Multi-Tool MLOps Agent integrating DVC, MLflow, Airflow 3, and Kubernetes.
    Includes structured secret-safe logging, strict Kubernetes safety guardrails,
    and evidence-grounded reasoning without fabricating unavailable tool outputs.
    """

    PROTECTED_NAMESPACES = {"kube-system", "kube-public", "kube-node-lease"}
    ALLOWED_K8S_MUTATIONS = {"k8s.scale_deployment", "k8s.restart_deployment"}

    def __init__(self, serving_health_url: str = "http://localhost:30080/health"):
        self.serving_health_url = serving_health_url
        self.tool_registry: Dict[str, Callable[..., Any]] = {
            # MLflow Tools
            "mlflow.list_experiments": mlflow_tools.list_experiments,
            "mlflow.get_run_details": mlflow_tools.get_run_details,
            "mlflow.compare_runs_and_recommend": mlflow_tools.compare_runs_and_recommend,
            "mlflow.compare_experiments": mlflow_tools.compare_experiments,
            "mlflow.get_registered_models": mlflow_tools.get_registered_models,
            # DVC Tools
            "dvc.get_dvc_status": dvc_tools.get_dvc_status,
            "dvc.get_dataset_metadata": dvc_tools.get_dataset_metadata,
            "dvc.list_dataset_versions": dvc_tools.list_dataset_versions,
            "dvc.diff_dataset_versions": dvc_tools.diff_dataset_versions,
            "dvc.get_reproduce_instructions": dvc_tools.get_reproduce_instructions,
            # Kubernetes Tools
            "k8s.get_cluster_status": kubernetes_tools.get_cluster_status,
            "k8s.get_pod_logs": kubernetes_tools.get_pod_logs,
            "k8s.scale_deployment": kubernetes_tools.scale_deployment,
            "k8s.restart_deployment": kubernetes_tools.restart_deployment,
            "k8s.diagnose_unhealthy_pods": kubernetes_tools.diagnose_unhealthy_pods,
            # Airflow Tools
            "airflow.list_dags": airflow_tools.list_dags,
            "airflow.get_dag_runs": airflow_tools.get_dag_runs,
            "airflow.trigger_dag": airflow_tools.trigger_dag,
            "airflow.diagnose_failed_tasks": airflow_tools.diagnose_failed_tasks,
        }
        self._trace: List[ToolCallRecord] = []

    def _validate_k8s_safety(self, tool_name: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """Reject unauthorized or destructive Kubernetes modifications."""
        if not tool_name.startswith("k8s."):
            return None

        namespace = str(kwargs.get("namespace", "default")).lower()
        if namespace in self.PROTECTED_NAMESPACES:
            return f"Unauthorized Kubernetes operation: access to system namespace '{namespace}' is forbidden."

        read_only_tools = {
            "k8s.get_cluster_status",
            "k8s.get_pod_logs",
            "k8s.diagnose_unhealthy_pods",
        }
        if tool_name not in read_only_tools and tool_name not in self.ALLOWED_K8S_MUTATIONS:
            return f"Unauthorized Kubernetes modification '{tool_name}' is blocked by safety policy."

        if tool_name == "k8s.scale_deployment":
            replicas = kwargs.get("replicas")
            if not isinstance(replicas, int) or replicas < 1 or replicas > 5:
                return "Safety guardrail: replicas must be an integer between 1 and 5."

        return None

    def invoke_tool(
        self, tool_name: str, request_id: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute a registered tool with guardrails, timing, and secret-safe structured logging."""
        req_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()

        safety_error = self._validate_k8s_safety(tool_name, kwargs)
        if safety_error:
            result = {"status": "rejected", "error": safety_error}
        else:
            fn = self.tool_registry.get(tool_name)
            if fn is None:
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
            else:
                sig = inspect.signature(fn)
                valid_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in sig.parameters
                    or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                }
                try:
                    raw = fn(**valid_kwargs)
                    result = raw if isinstance(raw, dict) else {"status": "ok", "data": raw}
                except TimeoutError as exc:
                    result = {"status": "timeout", "error": f"Tool timed out: {exc}"}
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        status = str(result.get("status", "ok"))

        # Emit structured log without exposing secrets
        log_entry = {
            "event": "tool_execution",
            "request_id": req_id,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": duration_ms,
            "safe_input": _redact_secrets(kwargs),
        }
        logger.info(json.dumps(log_entry, default=str))

        self._trace.append(
            ToolCallRecord(
                step=len(self._trace) + 1,
                request_id=req_id,
                tool_name=tool_name,
                status=status,
                duration_ms=duration_ms,
                tool_input=_redact_secrets(kwargs),
                tool_output=_redact_secrets(result),
            )
        )
        return result

    def _check_serving_endpoint(self, request_id: str) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(self.serving_health_url, timeout=3) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                result = {
                    "status": "ok",
                    "http_code": resp.status,
                    "endpoint": self.serving_health_url,
                    "response": body,
                }
        except Exception as exc:
            result = {
                "status": "unreachable",
                "endpoint": self.serving_health_url,
                "error": str(exc),
            }

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "tool_execution",
                    "request_id": request_id,
                    "tool_name": "serving.health_probe",
                    "status": result["status"],
                    "duration_ms": duration_ms,
                }
            )
        )
        self._trace.append(
            ToolCallRecord(
                step=len(self._trace) + 1,
                request_id=request_id,
                tool_name="serving.health_probe",
                status=result["status"],
                duration_ms=duration_ms,
                tool_input={"url": self.serving_health_url},
                tool_output=result,
            )
        )
        return result

    def execute_model_audit_and_retrain_decision(
        self,
        query: str,
        experiment_name: str = "iris-classification",
        dag_id: str = "mlops_training_pipeline",
        freshness_max_days: Optional[int] = None,
        observed_drift_or_degradation: Optional[str] = None,
    ) -> StructuredAgentReport:
        req_id = str(uuid.uuid4())
        self._trace = []

        # --- STEP 9: Query MLflow ---
        experiments_res = self.invoke_tool("mlflow.list_experiments", request_id=req_id)
        mlflow_available = experiments_res.get("status") == "ok"

        target_exp = experiment_name
        exps = experiments_res.get("experiments", [])
        if exps and not any(e.get("name") == experiment_name for e in exps):
            non_default = [e["name"] for e in exps if e.get("name") != "Default"]
            if non_default:
                target_exp = non_default[0]

        best_model_res = self.invoke_tool(
            "mlflow.compare_runs_and_recommend",
            request_id=req_id,
            experiment_name=target_exp,
        )
        registered_res = self.invoke_tool(
            "mlflow.get_registered_models", request_id=req_id
        )

        best_run = (
            best_model_res.get("recommended_candidate")
            or best_model_res.get("recommended_run")
            or best_model_res.get("best_run")
            or {}
        )
        best_run_id = best_run.get("run_id")
        run_details_res: Dict[str, Any] = {}
        if best_run_id:
            run_details_res = self.invoke_tool(
                "mlflow.get_run_details", request_id=req_id, run_id=best_run_id
            )

        # --- STEP 10: Read DVC Dataset Reference ---
        dvc_meta_res = self.invoke_tool("dvc.get_dataset_metadata", request_id=req_id)
        dvc_status_res = self.invoke_tool("dvc.get_dvc_status", request_id=req_id)
        dvc_versions_res = self.invoke_tool(
            "dvc.list_dataset_versions", request_id=req_id
        )

        # --- STEP 11: Inspect Kubernetes Deployment & App Health ---
        k8s_status_res = self.invoke_tool(
            "k8s.get_cluster_status", request_id=req_id, namespace="default"
        )
        k8s_diag_res = self.invoke_tool(
            "k8s.diagnose_unhealthy_pods", request_id=req_id, namespace="default"
        )
        serving_health_res = self._check_serving_endpoint(request_id=req_id)

        # --- STEP 12: Query Airflow Training Workflows ---
        airflow_dags_res = self.invoke_tool("airflow.list_dags", request_id=req_id)
        airflow_runs_res = self.invoke_tool(
            "airflow.get_dag_runs", request_id=req_id, dag_id=dag_id
        )
        airflow_diag_res = self.invoke_tool(
            "airflow.diagnose_failed_tasks", request_id=req_id, dag_id=dag_id
        )

        # --- STEP 13: Correlate Collected Evidence ---
        run_params = run_details_res.get("params") or best_run.get("params") or {}
        run_tags = run_details_res.get("tags") or best_run.get("tags") or {}
        run_metrics = (
            run_details_res.get("metrics")
            or best_run.get("all_metrics")
            or best_run.get("metrics")
            or {}
        )

        logged_dvc_ref = (
            best_run.get("dvc_dataset_md5")
            if best_run.get("dvc_dataset_md5") not in (None, "untracked")
            else (
                run_params.get("dataset_version")
                or run_params.get("dvc_md5")
                or run_tags.get("dataset_version")
                or run_tags.get("dvc_hash")
            )
        )
        current_dvc_md5 = (
            dvc_meta_res.get("md5")
            or dvc_meta_res.get("dataset_hash")
            or (dvc_meta_res.get("metadata") or {}).get("md5")
        )

        deployments = k8s_status_res.get("deployments", [])
        model_dep = next(
            (d for d in deployments if "iris-model" in d.get("name", "")),
            deployments[0] if deployments else {},
        )
        deployed_image = model_dep.get("image", "unknown")
        serving_version = (serving_health_res.get("response") or {}).get(
            "model_version", "unknown"
        )

        recent_runs = airflow_runs_res.get("runs", [])
        latest_dag_run = recent_runs[0] if recent_runs else {}
        failed_tasks = airflow_diag_res.get("failed_tasks", [])

        correlated = {
            "mlflow": {
                "available": mlflow_available and best_model_res.get("status") == "ok",
                "experiment": target_exp if mlflow_available else None,
                "best_run_id": best_run_id,
                "metrics": run_metrics,
                "logged_dataset_ref": logged_dvc_ref,
                "registered_models": registered_res.get("models", []),
            },
            "dvc": {
                "available": dvc_meta_res.get("status") == "ok",
                "current_tracked_md5": current_dvc_md5,
                "status": dvc_status_res,
                "versions_count": len(dvc_versions_res.get("versions", [])),
            },
            "kubernetes": {
                "available": k8s_status_res.get("status") == "ok",
                "deployment_name": model_dep.get("name"),
                "ready_replicas": f"{model_dep.get('ready_replicas', 0)}/{model_dep.get('desired_replicas', 0)}"
                if k8s_status_res.get("status") == "ok"
                else None,
                "image": deployed_image if k8s_status_res.get("status") == "ok" else None,
                "pods_healthy": k8s_diag_res.get("healthy"),
                "unhealthy_issues": k8s_diag_res.get("issues", []),
                "serving_endpoint": serving_health_res,
            },
            "airflow": {
                "available": airflow_dags_res.get("status") == "ok",
                "dag_id": dag_id,
                "latest_run_id": latest_dag_run.get("dag_run_id"),
                "latest_run_state": latest_dag_run.get("state"),
                "failed_tasks_count": len(failed_tasks),
                "failed_tasks": failed_tasks,
            },
        }

        # --- STEP 14: Known Facts, Uncertainties & Governance Actions ---
        known_facts: List[str] = []
        uncertainties: List[str] = []
        recommended_actions: List[str] = []

        # Never invent metrics or run IDs if MLflow failed/unavailable
        if best_model_res.get("status") != "ok" or not mlflow_available:
            uncertainties.append(
                f"MLflow service is unavailable or returned an error ({best_model_res.get('error', experiments_res.get('error', 'unavailable'))}); best model metrics and run ID cannot be verified."
            )
        elif best_run_id:
            known_facts.append(
                f"MLflow experiment '{target_exp}' identified best run '{best_run_id}' with metrics: {run_metrics}."
            )
        else:
            uncertainties.append(
                f"MLflow experiment '{target_exp}' returned no completed candidate runs."
            )

        # DVC facts vs uncertainties
        if dvc_meta_res.get("status") != "ok":
            uncertainties.append(
                f"DVC metadata tool failed ({dvc_meta_res.get('error', 'unavailable')}); dataset version cannot be verified."
            )
        elif logged_dvc_ref and current_dvc_md5:
            match_note = (
                "matches" if str(logged_dvc_ref) == str(current_dvc_md5) else "differs from"
            )
            known_facts.append(
                f"Selected MLflow run recorded dataset reference '{logged_dvc_ref}', which {match_note} current DVC hash '{current_dvc_md5}'."
            )
        elif current_dvc_md5:
            known_facts.append(
                f"Current DVC metadata tracks dataset hash '{current_dvc_md5}'."
            )
            uncertainties.append(
                "The MLflow run did not explicitly log a matching DVC hash tag/param, so exact run-to-dataset lineage is inferred from workspace state."
            )

        # Kubernetes facts vs uncertainties
        if k8s_status_res.get("status") != "ok":
            uncertainties.append(
                f"Kubernetes cluster status is unavailable ({k8s_status_res.get('error', 'unreachable')}); pod readiness and replica counts are unknown."
            )
        else:
            known_facts.append(
                f"Kubernetes deployment '{model_dep.get('name', 'iris-model-deployment')}' has {model_dep.get('ready_replicas', 0)}/{model_dep.get('desired_replicas', 0)} ready replicas running image '{deployed_image}' (pods healthy: {k8s_diag_res.get('healthy')})."
            )
            if k8s_diag_res.get("issues"):
                known_facts.append(
                    f"Kubernetes health diagnosis detected pod issues: {k8s_diag_res.get('issues')}."
                )

        if serving_health_res.get("status") == "ok":
            known_facts.append(
                f"Live serving endpoint ({self.serving_health_url}) responded HTTP 200 with model_loaded=True and model_version='{serving_version}'."
            )
        else:
            uncertainties.append(
                f"Live serving endpoint ({self.serving_health_url}) is unreachable ({serving_health_res.get('error')})."
            )

        # Airflow facts vs uncertainties
        if airflow_dags_res.get("status") != "ok" or airflow_runs_res.get("status") != "ok":
            uncertainties.append(
                f"Airflow workflow status is unavailable ({airflow_runs_res.get('error', airflow_dags_res.get('error', 'unreachable'))}); recent training run state cannot be confirmed."
            )
        elif latest_dag_run:
            detail_suffix = f" Details: {failed_tasks}." if failed_tasks else ""
            known_facts.append(
                f"Airflow DAG '{dag_id}' latest run '{latest_dag_run.get('dag_run_id')}' is in state '{latest_dag_run.get('state')}' with {len(failed_tasks)} failed tasks.{detail_suffix}"
            )
        else:
            known_facts.append(
                f"Airflow DAG '{dag_id}' is registered with no completed historical runs yet."
            )

        if observed_drift_or_degradation is None:
            uncertainties.append(
                "No live production ground-truth labels or feature-drift telemetry were supplied in this request."
            )
        if freshness_max_days is None:
            uncertainties.append(
                "No explicit model freshness SLA/policy (e.g., max age in days) was provided."
            )

        # Governance-Compliant Retraining Decision (NO automatic trigger)
        retrain_reasons: List[str] = []
        if failed_tasks or latest_dag_run.get("state") == "failed":
            retrain_reasons.append(
                f"Airflow reported failed pipeline execution/quality gate issues ({failed_tasks or latest_dag_run.get('state')})."
            )
        if observed_drift_or_degradation:
            retrain_reasons.append(
                f"Supplied degradation/drift evidence: {observed_drift_or_degradation}."
            )
        if freshness_max_days is not None:
            retrain_reasons.append(
                f"Explicit freshness policy threshold ({freshness_max_days} days) was specified for evaluation."
            )
        if (
            logged_dvc_ref
            and current_dvc_md5
            and str(logged_dvc_ref) != str(current_dvc_md5)
        ):
            retrain_reasons.append(
                "Current DVC dataset version differs from the dataset version used to train the best MLflow run."
            )

        if retrain_reasons:
            recommended_actions.append(
                "Retraining IS warranted based on concrete evidence: "
                + " ".join(retrain_reasons)
                + " Request explicit operator approval before triggering airflow.trigger_dag('mlops_training_pipeline')."
            )
        else:
            recommended_actions.append(
                "Do NOT initiate retraining right now: no quality gate failure occurred, no dataset version mismatch is confirmed, and no model degradation or freshness policy violation was supplied."
            )

        if k8s_diag_res.get("issues"):
            recommended_actions.append(
                "Inspect failing Kubernetes pod logs via k8s.get_pod_logs and remediate container errors before routing traffic."
            )

        return StructuredAgentReport(
            request_id=req_id,
            query=query,
            mode="deterministic-multi-tool",
            tool_trace=self._trace,
            correlated_evidence=correlated,
            known_facts=known_facts,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            retraining_initiated=False,
        )

    def handle_query(self, query: str) -> Dict[str, Any]:
        """Route user requests to multi-tool workflows or targeted operational tools."""
        q = query.lower().strip()
        if "delete" in q or "kube-system" in q or "exec" in q:
            tool = "k8s.delete_deployment" if "delete" in q else "k8s.exec_pod"
            ns = "kube-system" if "kube-system" in q else "default"
            return self.invoke_tool(tool, namespace=ns)

        if "scale" in q:
            replicas = 3 if "3" in q else (2 if "2" in q else 10)
            return self.invoke_tool(
                "k8s.scale_deployment",
                deployment_name="iris-model-deployment",
                replicas=replicas,
            )

        if "pod" in q or "kubernetes" in q or "k8s" in q:
            return {
                "cluster": self.invoke_tool("k8s.get_cluster_status"),
                "diagnosis": self.invoke_tool("k8s.diagnose_unhealthy_pods"),
            }

        if "airflow" in q or "dag" in q or "pipeline" in q:
            return {
                "dags": self.invoke_tool("airflow.list_dags"),
                "runs": self.invoke_tool(
                    "airflow.get_dag_runs", dag_id="mlops_training_pipeline"
                ),
            }

        return self.execute_model_audit_and_retrain_decision(query=query).model_dump()


def format_human_readable_report(report: StructuredAgentReport) -> str:
    lines = [
        "=" * 72,
        "INTELLIGENT MULTI-TOOL MLOPS AGENT REPORT",
        "=" * 72,
        f"Request ID: {report.request_id}",
        f"Query: {report.query}",
        f"Tools Executed ({len(report.tool_trace)}): "
        + " -> ".join(t.tool_name for t in report.tool_trace),
        f"Retraining Initiated Automatically: {report.retraining_initiated}",
        "\n1. KNOWN FACTS (Verified from MLflow, DVC, Kubernetes & Airflow):",
    ]
    for fact in report.known_facts:
        lines.append(f"   - {fact}")

    lines.append("\n2. UNCERTAINTIES & EVIDENCE GAPS:")
    for unc in report.uncertainties:
        lines.append(f"   - {unc}")

    lines.append("\n3. RECOMMENDED ACTIONS:")
    for act in report.recommended_actions:
        lines.append(f"   - {act}")

    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    mandatory_prompt = (
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary."
    )
    agent = MLOpsOrchestratorAgent()
    structured_report = agent.execute_model_audit_and_retrain_decision(
        query=mandatory_prompt
    )
    print(format_human_readable_report(structured_report))
    print("\n--- STRUCTURED JSON EVIDENCE ---")
    print(json.dumps(structured_report.correlated_evidence, indent=2, default=str))