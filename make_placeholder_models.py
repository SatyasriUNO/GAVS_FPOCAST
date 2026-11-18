"""
Creates small placeholder ONNX models so the whole pipeline can be tested
before the real crop models are trained.

Run once from the server folder:
    python3 make_placeholder_models.py

Later, replace server/models/<crop>/model.onnx with your trained model
(for YOLOv8:  yolo export model=best.pt format=onnx imgsz=640  -> rename to model.onnx)
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = [
    {"id": "potato_v1", "crop": "Potato", "icon": "🥔", "task": "Crop vision · plant count & canopy", "arch": "YOLOv8n-seg", "version": "1.0.0"},
    {"id": "tomato_v1", "crop": "Tomato", "icon": "🍅", "task": "Crop vision · fruit & canopy",       "arch": "YOLOv8n-seg", "version": "1.0.0"},
    {"id": "onion_v1",  "crop": "Onion",  "icon": "🧅", "task": "Crop vision · stand & canopy",       "arch": "YOLOv8n-seg", "version": "1.0.0"},
    {"id": "paddy_v1",  "crop": "Paddy",  "icon": "🌾", "task": "Crop vision · canopy & stress",      "arch": "YOLOv8n-seg", "version": "0.9.0"},
]

out_dir = Path(__file__).resolve().parent / "models"

for m in MODELS:
    folder = out_dir / m["id"]
    folder.mkdir(parents=True, exist_ok=True)

    # Tiny network: one 3x3 convolution on a 640x640 RGB image
    w = numpy_helper.from_array(np.random.rand(8, 3, 3, 3).astype(np.float32), "W")
    node = helper.make_node("Conv", ["images", "W"], ["output"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], m["id"],
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 640, 640])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, folder / "model.onnx")

    info = {k: v for k, v in m.items() if k != "id"}
    info["placeholder"] = True
    (folder / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("created", folder)
