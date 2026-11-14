import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def _run_cmd(cmd: list[str]) -> Dict[str, Any]:
    """Safely execute a CLI command and capture output."""
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "success": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "returncode": res.returncode,
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def get_dvc_status() -> Dict[str, Any]:
    """Check if tracked datasets or pipelines have local changes compared to DVC cache."""
    res = _run_cmd(["dvc", "status"])
    if not res["success"]:
        return {"status": "error", "error": res["stderr"]}
    output = res["stdout"] or "Data and pipelines are up to date."
    has_changes = "up to date" not in output.lower()
    return {
        "status": "ok",
        "has_changes": has_changes,
        "summary": output,
    }


def get_dataset_metadata(dvc_file: str = "data/dataset.csv.dvc") -> Dict[str, Any]:
    """Read current MD5 hash, size, and tracked path from a .dvc file."""
    path = Path(dvc_file)
    if not path.exists():
        return {"status": "error", "error": f"DVC file not found: {dvc_file}"}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        outs = data.get("outs", [])
        if not outs:
            return {"status": "error", "error": "No tracked outputs found in .dvc file."}
        entry = outs[0]
        return {
            "status": "ok",
            "dvc_file": dvc_file,
            "tracked_path": entry.get("path"),
            "md5": entry.get("md5"),
            "size_bytes": entry.get("size"),
            "nfiles": entry.get("nfiles", 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def list_dataset_versions(dvc_file: str = "data/dataset.csv.dvc") -> Dict[str, Any]:
    """List Git commit history and tags associated with a DVC-tracked dataset."""
    log_res = _run_cmd(
        ["git", "log", "--pretty=format:%H|%h|%ad|%s|%D", "--date=short", "--", dvc_file]
    )
    if not log_res["success"]:
        return {"status": "error", "error": log_res["stderr"]}

    versions = []
    for line in log_res["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 4)
        commit_hash = parts[0]
        short_hash = parts[1]
        date = parts[2]
        message = parts[3]
        refs = parts[4] if len(parts) > 4 else ""

        # Read the .dvc file content at that specific commit to extract the MD5 hash
        show_res = _run_cmd(["git", "show", f"{commit_hash}:{dvc_file}"])
        md5 = None
        size = None
        if show_res["success"]:
            parsed = yaml.safe_load(show_res["stdout"]) or {}
            outs = parsed.get("outs", [])
            if outs:
                md5 = outs[0].get("md5")
                size = outs[0].get("size")

        versions.append(
            {
                "commit": short_hash,
                "full_commit": commit_hash,
                "date": date,
                "message": message,
                "refs": refs,
                "md5": md5,
                "size_bytes": size,
            }
        )

    return {
        "status": "ok",
        "dvc_file": dvc_file,
        "total_versions": len(versions),
        "versions": versions,
    }


def diff_dataset_versions(rev_a: str = "data-v1", rev_b: str = "HEAD") -> Dict[str, Any]:
    """Compare DVC-tracked dataset changes between two Git revisions or tags."""
    res = _run_cmd(["dvc", "diff", rev_a, rev_b, "--json"])
    if not res["success"]:
        # Fallback to plain text dvc diff if --json fails on older/newer versions
        fallback = _run_cmd(["dvc", "diff", rev_a, rev_b])
        if not fallback["success"]:
            return {"status": "error", "error": fallback["stderr"] or res["stderr"]}
        return {"status": "ok", "rev_a": rev_a, "rev_b": rev_b, "diff": fallback["stdout"]}

    return {
        "status": "ok",
        "rev_a": rev_a,
        "rev_b": rev_b,
        "diff": res["stdout"],
    }


def get_reproduce_instructions(target_ref: str, dvc_file: str = "data/dataset.csv.dvc") -> Dict[str, Any]:
    """Provide safe, read-only CLI instructions to checkout/reproduce a specific dataset version."""
    return {
        "status": "ok",
        "target_ref": target_ref,
        "dvc_file": dvc_file,
        "commands": [
            f"git checkout {target_ref} -- {dvc_file}",
            f"dvc checkout {dvc_file}",
        ],
        "note": "Run these commands to restore the exact dataset state without modifying other repository files.",
    }