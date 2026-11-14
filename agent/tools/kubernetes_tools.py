from datetime import datetime, timezone
from typing import Dict, Any, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _load_k8s_clients():
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()
    return client.CoreV1Api(), client.AppsV1Api()


def get_cluster_status(namespace: str = "default") -> Dict[str, Any]:
    """List deployments and pods with their health and replica status."""
    try:
        core_v1, apps_v1 = _load_k8s_clients()
        deployments = apps_v1.list_namespaced_deployment(namespace=namespace).items
        pods = core_v1.list_namespaced_pod(namespace=namespace).items

        dep_list = [
            {
                "name": d.metadata.name,
                "desired_replicas": d.spec.replicas,
                "ready_replicas": d.status.ready_replicas or 0,
                "available_replicas": d.status.available_replicas or 0,
                "image": d.spec.template.spec.containers[0].image,
            }
            for d in deployments
        ]

        pod_list = []
        for p in pods:
            restarts = sum(
                (cs.restart_count for cs in (p.status.container_statuses or [])), 0
            )
            ready = all(
                (cs.ready for cs in (p.status.container_statuses or []))
            ) and bool(p.status.container_statuses)
            pod_list.append(
                {
                    "name": p.metadata.name,
                    "phase": p.status.phase,
                    "ready": ready,
                    "restarts": restarts,
                    "node": p.spec.node_name,
                }
            )

        return {
            "status": "ok",
            "namespace": namespace,
            "deployments": dep_list,
            "pods": pod_list,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_pod_logs(
    pod_name: str, namespace: str = "default", tail_lines: int = 50
) -> Dict[str, Any]:
    """Fetch recent container logs from a specific pod."""
    try:
        core_v1, _ = _load_k8s_clients()
        logs = core_v1.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, tail_lines=tail_lines
        )
        return {"status": "ok", "pod_name": pod_name, "logs": logs}
    except ApiException as e:
        return {"status": "error", "error": f"K8s API error: {e.reason}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def scale_deployment(
    deployment_name: str, replicas: int, namespace: str = "default"
) -> Dict[str, Any]:
    """Scale a deployment to the requested number of replicas (with safety guardrails)."""
    if replicas < 1 or replicas > 5:
        return {
            "status": "rejected",
            "error": "Safety guardrail: replicas must be between 1 and 5.",
        }
    try:
        _, apps_v1 = _load_k8s_clients()
        body = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name, namespace=namespace, body=body
        )
        return {
            "status": "ok",
            "deployment": deployment_name,
            "target_replicas": replicas,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def restart_deployment(
    deployment_name: str, namespace: str = "default"
) -> Dict[str, Any]:
    """Trigger a zero-downtime rolling restart of a deployment."""
    try:
        _, apps_v1 = _load_k8s_clients()
        now = datetime.now(timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {"kubectl.kubernetes.io/restartedAt": now}
                    }
                }
            }
        }
        apps_v1.patch_namespaced_deployment(
            name=deployment_name, namespace=namespace, body=body
        )
        return {
            "status": "ok",
            "deployment": deployment_name,
            "restarted_at": now,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def diagnose_unhealthy_pods(namespace: str = "default") -> Dict[str, Any]:
    """Inspect pods for CrashLoopBackOff, ImagePullBackOff, or readiness failures."""
    try:
        core_v1, _ = _load_k8s_clients()
        pods = core_v1.list_namespaced_pod(namespace=namespace).items
        issues = []

        for p in pods:
            pod_issues = []
            if p.status.phase not in ("Running", "Succeeded"):
                pod_issues.append(f"Pod phase is {p.status.phase}")

            for cs in p.status.container_statuses or []:
                if not cs.ready:
                    pod_issues.append(f"Container '{cs.name}' is not ready")
                if cs.restart_count > 0:
                    pod_issues.append(
                        f"Container '{cs.name}' restarted {cs.restart_count} times"
                    )
                if cs.state and cs.state.waiting:
                    pod_issues.append(
                        f"Waiting: {cs.state.waiting.reason} - {cs.state.waiting.message}"
                    )

            if pod_issues:
                issues.append({"pod_name": p.metadata.name, "reasons": pod_issues})

        return {
            "status": "ok",
            "healthy": len(issues) == 0,
            "unhealthy_count": len(issues),
            "issues": issues,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}