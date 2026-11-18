"""
FPOCAST 2.0 - Pi Agent
----------------------
Runs on the drone's Raspberry Pi 5.

What it does:
  1. POST /load   -> receives a model file from the ground server, then
                     verifies it (sha256), loads it (onnxruntime) and does a test run
  2. GET  /health -> tells the server which model is loaded and what stage it is in

Model files are saved in ~/fpocast_models/
"""

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

MODELS_DIR = Path.home() / "fpocast_models"
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="FPOCAST Pi Agent")

pi = {
    "stage": "idle",        # idle | verifying | loading | testing | ready | error
    "loaded_model": None,
    "error": None,
    "load_ms": None,
    "inference_ms": None,
}
lock = threading.Lock()

# The model currently in memory. Your survey/camera code uses this.
current_session = None


def set_state(**kw):
    with lock:
        pi.update(kw)


def load_worker(model_id: str, path: Path, expected_sha: str):
    global current_session
    try:
        # 1. Verify the file arrived complete
        set_state(stage="verifying")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError("Checksum mismatch - file damaged in transfer")

        # 2. Load into memory
        set_state(stage="loading")
        t0 = time.perf_counter()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        load_ms = round((time.perf_counter() - t0) * 1000)

        # 3. Test run on a dummy image (any unknown size -> 1 for batch, 640 otherwise)
        set_state(stage="testing")
        inp = session.get_inputs()[0]
        shape = [d if isinstance(d, int) else (1 if i == 0 else 640) for i, d in enumerate(inp.shape)]
        dummy = np.random.rand(*shape).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {inp.name: dummy})
        inf_ms = round((time.perf_counter() - t0) * 1000)

        current_session = session
        set_state(stage="ready", loaded_model=model_id, load_ms=load_ms, inference_ms=inf_ms, error=None)
        print(f"[pi_agent] {model_id} ready: load {load_ms} ms, test inference {inf_ms} ms")
    except Exception as e:
        set_state(stage="error", error=str(e))
        print(f"[pi_agent] ERROR loading {model_id}: {e}")


@app.get("/health")
def health():
    with lock:
        return dict(pi)


@app.post("/load", status_code=202)
async def load(model_id: str = Form(...), sha256: str = Form(...), file: UploadFile = File(...)):
    with lock:
        if pi["stage"] in ("verifying", "loading", "testing"):
            raise HTTPException(409, "Already loading a model")
    path = MODELS_DIR / f"{model_id}.onnx"
    path.write_bytes(await file.read())
    set_state(stage="verifying", error=None)
    threading.Thread(target=load_worker, args=(model_id, path, sha256), daemon=True).start()
    return {"received": model_id}
