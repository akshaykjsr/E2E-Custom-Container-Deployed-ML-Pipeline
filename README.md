# Vertex AI Custom Container E2E Pipeline

This repository contains a **production‑grade, end‑to‑end (E2E) machine‑learning pipeline** that trains and serves an XGBoost model on **Google Cloud Vertex AI** using **separate custom Docker images** for training and online prediction.

> **Why this repo?**
> Vertex AI’s managed pipelines hide a lot of boilerplate.
> This project shows every moving part—so you can debug, extend, or rip‑and‑replace any step without reverse‑engineering Google’s samples.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Pipeline Walk‑through](#pipeline-walk-through)
5. [Local Testing](#local-testing)
6. [Cleaning Up](#cleaning-up)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

---

## Architecture

```
┌───────────────────┐   Compile & Upload    ┌───────────────────────┐
│  Developer Laptop │  pipeline.json  ───▶  │ Vertex AI Pipelines   │
└───────┬───────────┘                      └───────────┬───────────┘
        │ Build & Push images                          │
        │ gcloud builds submit                         │
        ▼                                              ▼
┌─────────────────────┐              ┌────────────────────────────┐
│  Train Docker image │──training──▶│  CustomTrainingJob (GKE)    │
└─────────────────────┘              │ writes model to GCS bucket │
                                     └──────────────┬─────────────┘
                                                    │ upload_model
                                                    ▼
                                    ┌────────────────────────────┐
                                    │ Vertex AI Model Registry   │
                                    └──────────────┬─────────────┘
                                                   deploy_model
                                                    ▼
                                    ┌────────────────────────────┐
                                    │  Endpoint + Serving image  │
                                    └────────────────────────────┘
```

* **Training image** — `train_image/`
  Runs `train.py`, persists a `joblib` model to GCS.

* **Serving image** — `serve_image/`
  Lightweight Flask server (`score.py`) that loads the model and responds to `POST /predict`.

* **Pipeline** — `pipeline.py`
  KFP v2 DSL that stitches together **build → train → upload → deploy**.

---

## Directory Structure

```
project/
  gcp-vertexai-e2e-custom-model-pipeline/
    bootstrap.sh
    repoCreation-imageCreation.sh
    payload.json
    prototype_train.csv
    .python-version
    config.env
    notes
    pipeline.py
    requirements.txt
    train_image/
      pre_processor.py
      train.py
      Dockerfile
      requirements.txt
    local_artifacts/
      model.joblib
    serve_image/
      pre_processor.py
      Dockerfile
      score.py
      requirements.txt
      local_artifacts/
    __pycache__/
      pipeline.cpython-311.pyc
```

*Only the high‑level items are shown; see the repo for full contents.*

---

## Quick Start

```bash
# 0) Set your GCP project & region
gcloud config set project <YOUR_PROJECT_ID>
export REGION=us-central1

# 1) Clone the repo and cd into it
git clone https://github.com/<you>/gcp-vertexai-e2e-custom-model-pipeline.git
cd gcp-vertexai-e2e-custom-model-pipeline

# 2) Edit config.env  (bucket, artifact repo names, machine types, hyperparameters)
nano config.env

# 3) Build buckets + Artifact Repos + Docker images
./repoCreation-imageCreation.sh

# 4) Compile the KFP v2 pipeline to JSON
./bootstrap.sh           # generates pipeline.json

# 5) Submit the pipeline run (replace values as needed)
gcloud ai pipelines run \
  --project=$PROJECT_ID \
  --region=$REGION \
  --file=pipeline.json \
  --parameter-values=TAG=draft-parametrized
```

> **Tip:** Each script is **idempotent**; you can rerun them safely during iteration.

---

## Pipeline Walk‑through

| Step             | Component (DSL)           | Backing Image            | Key Code                        | Output                      |
| ---------------- | ------------------------- | ------------------------ | ------------------------------- | --------------------------- |
| **Build & Push** | *script*                  | —                        | `repoCreation-imageCreation.sh` | Images in Artifact Registry |
| **Train**        | `CustomTrainingJobOp`     | `train_image:Dockerfile` | `train_image/train.py`          | `model.joblib` in GCS       |
| **Upload Model** | `component(upload_model)` | vertex‑sdk               | inline in `pipeline.py`         | model resource id           |
| **Deploy**       | `component(deploy_model)` | vertex‑sdk               | inline in `pipeline.py`         | endpoint id                 |

Hyperparameters are wired via `config.env`; edit and re‑run the pipeline to sweep values.

---

## Local Testing

```bash
# Train locally
cd train_image
python train.py --train_csv=../prototype_train.csv --model_dir=./local_artifacts

# Serve locally
cd ../serve_image
MODEL_PATH=../train_image/local_artifacts/model.joblib python score.py &
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feature1": 1.5, "feature2": 3.1}] }'
```

---

## Cleaning Up

```bash
# Delete endpoints & models
gcloud ai endpoints list
gcloud ai endpoints delete <ENDPOINT_ID> --region=$REGION
gcloud ai models list
gcloud ai models delete <MODEL_ID> --region=$REGION

# Remove buckets & repos if you created dedicated ones
gcloud storage rm -r gs://$BUCKET
gcloud artifacts repositories delete $REPO_TRAIN --location=$REGION
gcloud artifacts repositories delete $REPO_SERVE --location=$REGION
```

---

## Troubleshooting

* **Container image fails to pull**
  Ensure `$REGION-docker.pkg.dev` is correct and Artifact Registry is enabled.

* **Pipeline stuck in `Queued`**
  Check IAM: Vertex AI Service Account needs `Artifact Registry Reader` and `Storage Object Admin`.

* **Model returns 404**
  Deploy step might have failed; look at Vertex AI > Endpoints > Logs.

---

## Contributing

Pull requests are welcome! Open an issue first to discuss major changes.

---

## License

[Apache‑2.0](LICENSE)

---

