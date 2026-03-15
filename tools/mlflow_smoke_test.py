import os
import socket
import tempfile
from pathlib import Path

import mlflow


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set")

    vm_name = socket.gethostname()
    experiment_name = "remote-mlflow-smoke-test"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"smoke-{vm_name}") as run:
        mlflow.log_param("vm_name", vm_name)
        mlflow.log_param("tracking_uri", tracking_uri)
        mlflow.log_metric("smoke_metric", 1.23)

        tmpdir = Path(tempfile.mkdtemp())
        artifact = tmpdir / "smoke.txt"
        artifact.write_text(f"Smoke test from {vm_name}\n", encoding="utf-8")
        mlflow.log_artifact(str(artifact))

        print("Run ID:", run.info.run_id)
        print("Experiment:", experiment_name)
        print("Tracking URI:", tracking_uri)

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
