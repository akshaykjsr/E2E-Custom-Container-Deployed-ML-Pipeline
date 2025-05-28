"""
Prototype prediction server
• readiness:   GET /healthz  (Vertex probe)  and GET /
• metadata:    GET /v1/endpoints/<endpoint_id>/deployedModels/<deployed_model_id>
• inference:   POST /predict   with JSON {instances:[{…}]}
• (optional)   POST /v1/endpoints/<endpoint_id>:predict
"""

from flask import Flask, request, jsonify
import os, joblib, gcsfs, numpy as np, pandas as pd
from threading import Lock

print("Score Server Started")
app = Flask(__name__)
print("Starting prediction server")

_bundle = None          # {model, scaler, features}
_lock   = Lock()


# ---------------------- Vertex metadata probe -------------------------------
@app.route(
    "/v1/endpoints/<endpoint_id>/deployedModels/<deployed_model_id>",
    methods=["GET"]
)
def _get_deployed_model(endpoint_id, deployed_model_id):
    # Vertex sidecar polls this to see "is my model here yet?"
    return "", 200


# ---------------------- (Optional) Vertex predict wrapper ------------------
@app.route("/v1/endpoints/<endpoint_id>:predict", methods=["POST"])
def _predict_v1(endpoint_id):
    # delegate to the same logic as /predict
    return predict()


# ---------------------- lazy load artifact ----------------------------------
def get_bundle():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                artefact_dir = os.environ["AIP_STORAGE_URI"].rstrip("/")
                local_model = "/tmp/model.joblib"
                if artefact_dir.startswith("gs://"):
                    fs = gcsfs.GCSFileSystem()
                    fs.get(f"{artefact_dir}/model.joblib", local_model)
                else:
                    # local path: copy straight from disk
                    import shutil, pathlib
                    src = pathlib.Path(artefact_dir) / "model.joblib"
                    if not src.exists():
                        raise FileNotFoundError(f"Model file not found at {src}")
                    shutil.copy(src, local_model)
                _bundle = joblib.load(local_model)
    return _bundle


# ---------------------- health routes ---------------------------------------
@app.route("/healthz", methods=["GET"])
@app.route("/", methods=["GET"])
def health():
    try:
        bundle = _bundle or get_bundle()
        if bundle and "model" in bundle:
            return "ok", 200
    except Exception:
        pass
    return "not ready", 503

# ---------------------- prediction ------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "instances" not in body:
        return jsonify(error="JSON must contain key 'instances'"), 400

    inst_df = pd.DataFrame(body["instances"])

    bundle   = get_bundle()
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features"]

    # ensure all expected columns are present
    missing = [c for c in features if c not in inst_df.columns]
    if missing:
        return jsonify(error=f"Missing columns: {missing}"), 400

    X_scaled = scaler.transform(inst_df[features])
    y_pred   = model.predict(X_scaled)
    y_pred   = np.expm1(y_pred)          # reverse log1p

    return jsonify(predictions=y_pred.tolist()), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

