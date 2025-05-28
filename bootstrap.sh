#!/usr/bin/env bash
set -euo pipefail

# ───── 1. Config ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"



# ───── 2. Compile the KFP v2 pipeline to JSON ───────────────────────────────
python - <<'PY'
from kfp import compiler
from pipeline import pipeline

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="pipeline.json",
)
print("Compiled pipeline.json")
PY

# ───── 3. Submit the pipeline via the Python SDK ────────────────────────────
python - <<PY
import os
from datetime import datetime
from google.cloud import aiplatform

# Initialize the SDK
aiplatform.init(project="${PROJECT_ID}", location="${REGION}")

# Create and run the PipelineJob
pipeline_job = aiplatform.PipelineJob(
    display_name=f"train-upload-deploy-{datetime.utcnow():%Y%m%d%H%M%S}",
    template_path="pipeline.json",
    pipeline_root="gs://${BUCKET}/artifacts",
parameter_values={
    "project_id": "${PROJECT_ID}",
    "region": "${REGION}",
    "train_image_uri": "${TRAIN_IMG}",
    "serve_image_uri": "${SERVE_IMG}",
    "train_data_gcs": "gs://${BUCKET}/prototype_train.csv",
    "artifact_bucket": "gs://${BUCKET}/artifacts",
    "model_display_name": "${MODEL_DISPLAY_NAME}",
    "endpoint_display_name": "${ENDPOINT_DISPLAY_NAME}",
    "machine_type": "${MACHINE_TYPE}",
    "train_split": "${TRAIN_SPLIT}",
    "learning_rate": "${LEARNING_RATE}",
    "n_estimators": "${N_ESTIMATORS}",
    "random_state": "${RANDOM_STATE}",
    },
)

pipeline_job.run(sync=False)
print("Started pipeline:", pipeline_job.resource_name)
PY

