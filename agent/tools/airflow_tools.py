import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "nh32Cg3kEMukvNsm")
APPROVED_DAGS = {"mlops_training_pipeline"}


def _get_headers() -> Dict[str, str]:
    """Authenticate against Airflow 3 /auth/token endpoint and return Bearer headers."""
    token_url = f"{AIRFLOW_BASE_URL.rstrip('/')}/auth/token"
    resp = requests.post(
        token_url,
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Airflow auth succeeded but no access_token was returned.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def list_dags() -> Dict[str, Any]:
    """List available DAGs in Apache Airflow 3."""
    try:
        headers = _get_headers()
        url = f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v2/dags"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        dags = resp.json().get("dags", [])
        return {
            "status": "ok",
            "count": len(dags),
            "dags": [
                {
                    "dag_id": d.get("dag_id"),
                    "is_paused": d.get("is_paused"),
                    "description": d.get("description"),
                    "tags": [t.get("name") for t in d.get("tags", [])],
                    "approved_for_trigger": d.get("dag_id") in APPROVED_DAGS,
                }
                for d in dags
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_dag_runs(dag_id: str = "mlops_training_pipeline", limit: int = 5) -> Dict[str, Any]:
    """Retrieve recent DAG runs and their execution states."""
    try:
        headers = _get_headers()
        url = f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v2/dags/{dag_id}/dagRuns"
        resp = requests.get(
            url,
            headers=headers,
            params={"limit": limit, "order_by": "-logical_date"},
            timeout=10,
        )
        resp.raise_for_status()
        runs = resp.json().get("dag_runs", [])
        return {
            "status": "ok",
            "dag_id": dag_id,
            "count": len(runs),
            "runs": [
                {
                    "dag_run_id": r.get("dag_run_id"),
                    "state": r.get("state"),
                    "logical_date": r.get("logical_date"),
                    "start_date": r.get("start_date"),
                    "end_date": r.get("end_date"),
                    "conf": r.get("conf", {}),
                }
                for r in runs
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def trigger_dag(
    dag_id: str = "mlops_training_pipeline",
    conf: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Trigger an approved training DAG via Airflow 3 REST API."""
    if dag_id not in APPROVED_DAGS:
        return {
            "status": "error",
            "error": f"DAG '{dag_id}' is not in the approved list: {sorted(APPROVED_DAGS)}",
        }
    try:
        headers = _get_headers()
        base = AIRFLOW_BASE_URL.rstrip("/")

        # Unpause DAG if currently paused so execution starts immediately
        requests.patch(
            f"{base}/api/v2/dags/{dag_id}",
            headers=headers,
            json={"is_paused": False},
            timeout=10,
        )

        logical_date = datetime.now(timezone.utc).isoformat()
        resp = requests.post(
            f"{base}/api/v2/dags/{dag_id}/dagRuns",
            headers=headers,
            json={"logical_date": logical_date, "conf": conf or {}},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": "ok",
            "dag_id": dag_id,
            "dag_run_id": data.get("dag_run_id"),
            "state": data.get("state"),
            "logical_date": data.get("logical_date"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def diagnose_failed_tasks(
    dag_id: str = "mlops_training_pipeline",
    dag_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Identify failed task instances in a DAG run and retrieve execution logs/diagnostics."""
    try:
        headers = _get_headers()
        base = AIRFLOW_BASE_URL.rstrip("/")

        # If no dag_run_id supplied, pick the most recent run
        if not dag_run_id:
            runs_res = get_dag_runs(dag_id=dag_id, limit=1)
            if runs_res.get("status") != "ok" or not runs_res.get("runs"):
                return {"status": "error", "error": "No DAG runs found to inspect."}
            dag_run_id = runs_res["runs"][0]["dag_run_id"]

        ti_url = f"{base}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        ti_resp = requests.get(ti_url, headers=headers, timeout=10)
        ti_resp.raise_for_status()
        task_instances = ti_resp.json().get("task_instances", [])

        failed_details: List[Dict[str, Any]] = []
        all_tasks_summary = []

        for ti in task_instances:
            task_id = ti.get("task_id")
            state = ti.get("state")
            try_number = ti.get("try_number", 1)
            all_tasks_summary.append(
                {
                    "task_id": task_id,
                    "state": state,
                    "try_number": try_number,
                    "duration": ti.get("duration"),
                }
            )

            if state in {"failed", "upstream_failed"}:
                log_excerpt = ""
                if state == "failed":
                    log_url = f"{base}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{max(1, try_number)}"
                    log_resp = requests.get(
                        log_url,
                        headers={**headers, "Accept": "application/json"},
                        timeout=10,
                    )
                    if log_resp.ok:
                        content = log_resp.text
                        # Keep the last 1200 chars where the traceback/error appears
                        log_excerpt = content[-1200:]
                failed_details.append(
                    {
                        "task_id": task_id,
                        "state": state,
                        "try_number": try_number,
                        "start_date": ti.get("start_date"),
                        "end_date": ti.get("end_date"),
                        "log_excerpt": log_excerpt or "Blocked due to upstream task failure.",
                    }
                )

        return {
            "status": "ok",
            "dag_id": dag_id,
            "dag_run_id": dag_run_id,
            "tasks": all_tasks_summary,
            "failed_count": len(failed_details),
            "failed_tasks": failed_details,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
        