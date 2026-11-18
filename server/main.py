"""
FPOCAST 2.0 - Ground Server (Model Selector)
--------------------------------------------
Runs on the laptop / ground station.

What it does:
  1. Serves the web app (../web/index.html) at http://<this-ip>:8000
  2. GET  /models       -> list of crop models found in ./models/
  3. POST /models/load  -> sends the chosen model file to the Raspberry Pi
  4. GET  /status       -> server + Pi status, and progress of the current load

Each model lives in its own folder:
  server/models/potato_v1/info.json
  server/models/potato_v1/model.onnx
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR.parent / "web"

# Address of the Pi agent. Change with:  export PI_URL=http://<pi-ip>:9000
PI_URL = os.environ.get("PI_URL", "http://192.168.4.2:9000").rstrip("/")

app = FastAPI(title="FPOCAST Ground Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Progress of the current "load model" job
job = {
    "state": "idle",      # idle | sending | waiting | ready | error
    "model_id": None,
    "error": None,
    "started_at": None,
    "load_time_s": None,
}
job_lock = threading.Lock()


# ---------- helpers ----------

def read_models():
    """Scan server/models/*/info.json and return a list of model descriptions."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for folder in sorted(MODELS_DIR.iterdir()):
        info_file = folder / "info.json"
        model_file = folder / "model.onnx"
        if not (info_file.exists() and model_file.exists()):
            continue
        info = json.loads(info_file.read_text())
        info["id"] = folder.name
        info["size_mb"] = round(model_file.stat().st_size / 1_000_000, 2)
        models.append(info)
    return models


def pi_health():
    """Ask the Pi how it is doing. Returns {"online": False} if it does not answer."""
    try:
        r = requests.get(f"{PI_URL}/health", timeout=1.5)
        r.raise_for_status()
        data = r.json()
        data["online"] = True
        return data
    except Exception:
        return {"online": False}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_job(model_id: str):
    """Runs in a background thread: send model to Pi, then wait until Pi says ready."""
    model_file = MODELS_DIR / model_id / "model.onnx"
    try:
        # Step 1: send the file
        with open(model_file, "rb") as f:
            r = requests.post(
                f"{PI_URL}/load",
                data={"model_id": model_id, "sha256": sha256_of(model_file)},
                files={"file": (f"{model_id}.onnx", f, "application/octet-stream")},
                timeout=300,
            )
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", f"Pi returned HTTP {r.status_code}"))

        # Step 2: wait for the Pi to verify, load and test the model
        with job_lock:
            job["state"] = "waiting"
        deadline = time.time() + 300
        while time.time() < deadline:
            h = pi_health()
            if h.get("stage") == "ready" and h.get("loaded_model") == model_id:
                with job_lock:
                    job["state"] = "ready"
                    job["load_time_s"] = round(time.time() - job["started_at"], 1)
                return
            if h.get("stage") == "error":
                raise RuntimeError(h.get("error") or "Pi reported an error")
            time.sleep(0.4)
        raise RuntimeError("Timed out waiting for the Pi")
    except Exception as e:
        with job_lock:
            job["state"] = "error"
            job["error"] = str(e)


# ---------- API ----------

class LoadRequest(BaseModel):
    model_id: str


@app.get("/models")
def list_models():
    return read_models()


@app.post("/models/load", status_code=202)
def load_model(req: LoadRequest):
    ids = [m["id"] for m in read_models()]
    if req.model_id not in ids:
        raise HTTPException(404, f"Unknown model '{req.model_id}'")
    with job_lock:
        if job["state"] in ("sending", "waiting"):
            raise HTTPException(409, "A model is already being loaded")
        job.update(state="sending", model_id=req.model_id, error=None,
                   started_at=time.time(), load_time_s=None)
    threading.Thread(target=load_job, args=(req.model_id,), daemon=True).start()
    return {"job": "accepted", "model_id": req.model_id}


@app.get("/status")
def status():
    pi = pi_health()
    with job_lock:
        j = dict(job)
    # One simple "stage" word for the app to show
    if j["state"] == "waiting":
        stage = pi.get("stage", "waiting")   # verifying | loading | testing
        if stage == "ready":                 # server hasn't recorded it yet
            stage = "testing"
    else:
        stage = j["state"]                   # idle | sending | ready | error
    return {
        "server": "online",
        "pi": pi,
        "job": {**j, "stage": stage},
    }


@app.get("/")
def web_app():
    return FileResponse(WEB_DIR / "index.html")
