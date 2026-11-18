# FPOCAST 2.0 – Model Selector

Pick a crop model on your phone → the ground server sends it to the drone's Raspberry Pi → the Pi loads it and confirms.

App Interface Look & Feel - https://fpocastapp.vercel.app/
Website containing all information and initial data - 

```
Phone browser  ──►  Ground server (laptop, port 8000)  ──►  Raspberry Pi 5 (port 9000)
  web/index.html       server/main.py                          pi/pi_agent.py
```

## Folder structure

```
fpocast-app/
├── README.md
├── server/                      ← runs on the laptop / ground station
│   ├── main.py                  ← FastAPI server + serves the web page
│   ├── make_placeholder_models.py
│   ├── requirements.txt
│   └── models/                  ← one folder per crop model
│       ├── potato_v1/  (info.json + model.onnx)
│       ├── tomato_v1/
│       ├── onion_v1/
│       └── paddy_v1/
├── pi/                          ← copy this folder to the Raspberry Pi
│   ├── pi_agent.py
│   └── requirements.txt
└── web/
    └── index.html               ← the 3-screen app (Connect → Select → Status) 
```

---

## Step 1 – Laptop (ground server)

```bash
cd ~/fpocast-app/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 make_placeholder_models.py
```

Find the Pi's IP address (run this **on the Pi**): `hostname -I`

Start the server (replace `192.168.4.2` with the Pi's IP):

```bash
cd ~/fpocast-app/server
source venv/bin/activate
export PI_URL=http://192.168.4.2:9000
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 2 – Raspberry Pi 5

Copy the `pi` folder from the laptop to the Pi (replace user and IP):

```bash
scp -r ~/fpocast-app/pi pi@192.168.4.2:~/fpocast-pi
```

Then on the Pi:

```bash
ssh pi@192.168.4.2
cd ~/fpocast-pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn pi_agent:app --host 0.0.0.0 --port 9000
```

Loaded models are saved on the Pi in `~/fpocast_models/`.

## Step 3 – Phone

1. Connect the phone to the same Wi-Fi as the laptop and Pi.
2. Find the laptop's IP: run `hostname -I` on the laptop.
3. Open `http://<laptop-ip>:8000` in the phone's browser.
4. Connect → pick a crop → Load onto drone.

---

## Adding a real trained model

1. Export your trained YOLOv8 model to ONNX (on the laptop):
   ```bash
   pip install ultralytics
   yolo export model=best.pt format=onnx imgsz=640
   ```
2. Make a folder and copy it in (example: potato):
   ```bash
   mkdir -p ~/fpocast-app/server/models/potato_v2
   cp best.onnx ~/fpocast-app/server/models/potato_v2/model.onnx
   nano ~/fpocast-app/server/models/potato_v2/info.json
   ```
3. Paste this into `info.json`, save with `Ctrl+O`, `Enter`, exit with `Ctrl+X`:
   ```json
   {
     "crop": "Potato",
     "icon": "🥔",
     "task": "Crop vision · plant count & canopy",
     "arch": "YOLOv8n-seg",
     "version": "2.0.0"
   }
   ```
4. Reload the page — the new model appears in the list. No code changes needed.

## API (for testing from the terminal)

```bash
curl http://localhost:8000/status
curl http://localhost:8000/models
curl -X POST http://localhost:8000/models/load -H 'Content-Type: application/json' -d '{"model_id":"potato_v1"}'
```

## Notes

- The 4 models from `make_placeholder_models.py` are **placeholders** (tiny test networks) so the pipeline can be shown before real crop models are trained. The app marks them with a "placeholder" tag.
- GPS and battery show "—" for now. They can be filled later from the Pixhawk over MAVLink in `pi_agent.py` → `/health`.
- Next step: wrap `web/index.html` into an Android app (Flutter WebView or Capacitor) — the server and Pi code stay the same.
