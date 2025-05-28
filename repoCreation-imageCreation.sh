#!/usr/bin/env bash
set -euo pipefail

# ───── 0. Config ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"


gcloud storage buckets create gs://${BUCKET} \
  --location=us-central1 \
  --uniform-bucket-level-access
gcloud storage cp prototype_train.csv gs://${BUCKET}/


# ───── 1. Build & push both images (skip if already done) ───────────────────
gcloud artifacts repositories create "${REPO_TRAIN}" --repository-format=docker --location="${REGION}" || true
gcloud artifacts repositories create "${REPO_SERVE}" --repository-format=docker --location="${REGION}" || true

gcloud builds submit train_image --tag "${TRAIN_IMG}" &
PID_TRAIN=$!

gcloud builds submit serve_image --tag "${SERVE_IMG}" &
PID_SERVE=$!

#wait $PID_TRAIN
wait $PID_SERVE
