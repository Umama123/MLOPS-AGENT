import os
from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient


def _get_client() -> MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def list_experiments() -> Dict[str, Any]:
    """List all active MLflow experiments."""
    try:
        client = _get_client()
        exps = client.search_experiments()
        results = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in exps
        ]
        return {"status": "ok", "count": len(results), "experiments": results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_run_details(run_id: str) -> Dict[str, Any]:
    """Retrieve full metrics, parameters, tags, and artifact list for a specific run."""
    try:
        client = _get_client()
        run = client.get_run(run_id)
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        return {
            "status": "ok",
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "run_status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": {
                k: v
                for k, v in run.data.tags.items()
                if not k.startswith("mlflow.")
            },
            "artifacts": artifacts,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def compare_runs_and_recommend(
    metric: str = "f1_score",
    limit: int = 5,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare recent training runs on a target metric and evaluate production readiness
    (checking overfitting gap, inference latency, and model registry status rather than
    blindly picking the highest raw metric).
    """
    try:
        client = _get_client()
        if experiment_name:
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                return {"status": "error", "error": f"Experiment '{experiment_name}' not found."}
            exp_ids = [exp.experiment_id]
        else:
            exp_ids = [e.experiment_id for e in client.search_experiments()]

        runs = client.search_runs(
            experiment_ids=exp_ids,
            order_by=["start_time DESC"],
            max_results=limit,
        )
        if not runs:
            return {"status": "error", "error": "No training runs found."}

        # Map run_id -> registered model versions if any
        registered_map: Dict[str, List[str]] = {}
        try:
            for rm in client.search_registered_models():
                for mv in client.search_model_versions(f"name='{rm.name}'"):
                    registered_map.setdefault(mv.run_id, []).append(f"{rm.name}:v{mv.version}")
        except Exception:
            pass

        compared = []
        for r in runs:
            m = r.data.metrics
            p = r.data.params
            tags = r.data.tags
            val = m.get(metric)
            overfit_gap = m.get("overfit_gap", 0.0)
            latency_ms = m.get("inference_latency_ms", 0.0)
            reg_versions = registered_map.get(r.info.run_id, [])

            # Production readiness warnings
            warnings = []
            if val == 1.0 and overfit_gap == 0.0:
                warnings.append("Suspiciously perfect score (1.0) — verify against data leakage or small test split.")
            if abs(overfit_gap) > 0.05:
                warnings.append(f"High train/test gap ({overfit_gap}) indicates potential overfitting/underfitting.")
            if latency_ms > 5.0:
                warnings.append(f"Elevated inference latency ({latency_ms} ms/sample).")
            if not reg_versions:
                warnings.append("Not yet registered in MLflow Model Registry.")

            compared.append(
                {
                    "run_id": r.info.run_id,
                    "short_id": r.info.run_id[:8],
                    "run_name": r.info.run_name,
                    "experiment_id": r.info.experiment_id,
                    "model_type": p.get("model_type", "unknown"),
                    "params": p,
                    "target_metric": metric,
                    "target_metric_value": val,
                    "all_metrics": m,
                    "dvc_dataset_md5": tags.get("dvc_dataset_md5", "untracked"),
                    "registered_versions": reg_versions,
                    "production_readiness_warnings": warnings,
                }
            )

        valid_runs = [c for c in compared if c["target_metric_value"] is not None]
        if not valid_runs:
            return {
                "status": "ok",
                "message": f"Runs found, but none recorded metric '{metric}'.",
                "runs": compared,
            }

        # Sort by target metric descending
        sorted_by_metric = sorted(valid_runs, key=lambda x: x["target_metric_value"], reverse=True)
        highest_metric_run = sorted_by_metric[0]

        # Prefer registered + low-warning candidate when selecting production candidate
        prod_candidates = sorted(
            valid_runs,
            key=lambda x: (
                len(x["registered_versions"]) > 0,
                len(x["production_readiness_warnings"]) == 0,
                x["target_metric_value"],
            ),
            reverse=True,
        )
        recommended = prod_candidates[0]

        return {
            "status": "ok",
            "metric": metric,
            "total_compared": len(compared),
            "highest_raw_metric_run": highest_metric_run["short_id"],
            "recommended_candidate": recommended,
            "selection_reasoning": (
                f"Selected run {recommended['short_id']} ({recommended['model_type']}) with {metric}="
                f"{recommended['target_metric_value']}. Highest raw score alone does not guarantee "
                "production readiness; overfitting gap, inference latency, DVC data lineage, and "
                "Model Registry status were evaluated."
            ),
            "runs": compared,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def compare_experiments(exp_a: str, exp_b: str, metric: str = "f1_score") -> Dict[str, Any]:
    """Compare summary performance across two named MLflow experiments."""
    res_a = compare_runs_and_recommend(metric=metric, limit=20, experiment_name=exp_a)
    res_b = compare_runs_and_recommend(metric=metric, limit=20, experiment_name=exp_b)
    if res_a.get("status") != "ok":
        return res_a
    if res_b.get("status") != "ok":
        return res_b

    def _summarize(exp_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        vals = [r["target_metric_value"] for r in data["runs"] if r["target_metric_value"] is not None]
        return {
            "experiment": exp_name,
            "run_count": len(data["runs"]),
            "best_metric": max(vals) if vals else None,
            "avg_metric": round(sum(vals) / len(vals), 4) if vals else None,
            "recommended_run": data["recommended_candidate"]["short_id"],
        }

    return {
        "status": "ok",
        "metric": metric,
        "comparison": [
            _summarize(exp_a, res_a),
            _summarize(exp_b, res_b),
        ],
    }


def get_registered_models() -> Dict[str, Any]:
    """Retrieve all registered models and their version details from the MLflow Registry."""
    try:
        client = _get_client()
        models = client.search_registered_models()
        output = []
        for rm in models:
            versions = client.search_model_versions(f"name='{rm.name}'")
            output.append(
                {
                    "name": rm.name,
                    "description": rm.description,
                    "versions": [
                        {
                            "version": v.version,
                            "run_id": v.run_id,
                            "status": v.status,
                            "aliases": getattr(v, "aliases", []),
                            "source": v.source,
                        }
                        for v in versions
                    ],
                }
            )
        return {"status": "ok", "count": len(output), "models": output}
    except Exception as e:
        return {"status": "error", "error": str(e)}
        