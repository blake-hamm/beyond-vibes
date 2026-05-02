"""Inspect MLflow traces for a given run."""

import json
import logging
import sys
import tempfile
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("https://mlflow.bhamm-lab.com")

run_id = sys.argv[1] if len(sys.argv) > 1 else "feb6b66ad43242d8a58db6c1f1e8783a"

client = mlflow.MlflowClient()
run = client.get_run(run_id)

logger.info("Run ID: %s", run.info.run_id)
logger.info("Status: %s", run.info.status)
logger.info("Metrics: %s", dict(run.data.metrics))
logger.info("Tags: %s", dict(run.data.tags))

# Get trace
experiment = client.get_experiment(run.info.experiment_id)
logger.info("Experiment: %s", experiment.name)

# List artifacts
try:
    artifacts = client.list_artifacts(run_id)
    logger.info("Artifacts (%d):", len(artifacts))
    for a in artifacts:
        logger.info("  %s (%d bytes)", a.path, a.file_size or 0)
except Exception as e:
    logger.error("Error listing artifacts: %s", e)

# Try to get trace data
logger.info("Attempting to load trace_session.json")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, "trace_session.json", tmpdir)
        data = json.loads(Path(path).read_text())
        logger.info("%s", json.dumps(data, indent=2)[:4000])
except Exception as e:
    logger.error("Error: %s", e)
