"""
Step 3 of 3 - Start the website.

What this script does, in plain English:
  - Starts a tiny web server on http://localhost:8000
  - Serves the HTML pages inside the web/ folder
  - Adds ONE extra endpoint: POST /predict
        - receives an uploaded image
        - loads the trained CNN (tomato_model.h5)
        - returns JSON like {"label": "rotten", "confidence": 92}
  - That's it. No frameworks. Uses only Python's built-in http.server
    plus TensorFlow (which you already installed for training).

Usage:
    python 3_server.py
Then open the browser at:
    http://localhost:8000/
"""

import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

# Hide TensorFlow's startup logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# --- Configuration --------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR      = PROJECT_ROOT / "web"
ASSETS_DIR   = PROJECT_ROOT / "assets"
DATASET_DIR  = PROJECT_ROOT / "dataset"
MODEL_PATH   = PROJECT_ROOT / "tomato_model.h5"
RESULTS_PATH = PROJECT_ROOT / "training_results.json"

HOST = "127.0.0.1"
PORT = 8000


# --- Load the trained model once, at startup ------------------------------- #

if not MODEL_PATH.exists():
    print("[!!] tomato_model.h5 not found.")
    print("     Run step 2 first:   python 2_train_model.py")
    sys.exit(1)

print(f"[..] Loading model {MODEL_PATH.name} ...")
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Auto-detect the input size from the model so this server works with whatever
# size 2_train_model.py was set to (default 128, but you can change it).
IMG_SIZE = MODEL.input_shape[1] or 128
print(f"[ok] Model ready (input size: {IMG_SIZE} x {IMG_SIZE}).")


# --- Helper: image bytes -> prediction ------------------------------------- #

def predict_from_bytes(image_bytes: bytes) -> dict:
    """Run the CNN on an uploaded image. Returns {label, confidence, percentages}."""
    img = tf.keras.utils.load_img(
        BytesIO(image_bytes), target_size=(IMG_SIZE, IMG_SIZE)
    )
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)         # shape: (1, H, W, 3)

    prob_rotten = float(MODEL.predict(arr, verbose=0).ravel()[0])

    is_rotten = prob_rotten >= 0.5
    # Confidence = distance from the 0.5 decision line, mapped to 50-100%.
    confidence = int(round(50 + abs(prob_rotten - 0.5) * 100))

    return {
        "label":          "rotten" if is_rotten else "fresh",
        "rotten_percent": round(prob_rotten * 100, 1),
        "fresh_percent":  round((1 - prob_rotten) * 100, 1),
        "confidence":     confidence,
    }


# --- HTTP handler ---------------------------------------------------------- #

class TomatoHandler(BaseHTTPRequestHandler):
    # --- response helpers --- #
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Not found")
            return

        mime = {
            ".html": "text/html; charset=utf-8",
            ".css":  "text/css; charset=utf-8",
            ".js":   "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg":  "image/svg+xml",
        }.get(file_path.suffix.lower(), "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stdout.write("  " + (fmt % args) + "\n")

    # --- routing --- #
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            self._send_file(WEB_DIR / "index.html")
            return
        if path == "/training_results.json":
            self._send_file(RESULTS_PATH)
            return
        if path.startswith("/assets/"):
            self._send_file(ASSETS_DIR / path[len("/assets/"):])
            return
        if path.startswith("/dataset/"):
            rel = path[len("/dataset/"):]
            requested = (DATASET_DIR / rel).resolve()
            if DATASET_DIR.resolve() in requested.parents:
                self._send_file(requested)
            else:
                self.send_error(403, "Forbidden")
            return
        # Anything else -> look under web/
        self._send_file(WEB_DIR / path.lstrip("/"))

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404, "Unknown POST endpoint")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 10 * 1024 * 1024:   # 10 MB max
            self._send_json(400, {"error": "Empty or too-large upload (max 10 MB)."})
            return

        raw = self.rfile.read(length)

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(400, {"error": "Expected multipart/form-data upload."})
            return

        try:
            boundary = content_type.split("boundary=", 1)[1].encode()
        except IndexError:
            self._send_json(400, {"error": "Missing multipart boundary."})
            return

        # Pull the file bytes out of the multipart body.
        try:
            parts = raw.split(b"--" + boundary)
            image_bytes = None
            for part in parts:
                if b"Content-Type: image/" in part:
                    image_bytes = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n--")
                    break
            if not image_bytes:
                self._send_json(400, {"error": "No image found in upload."})
                return
        except Exception as e:
            self._send_json(400, {"error": f"Could not parse upload: {e}"})
            return

        try:
            result = predict_from_bytes(image_bytes)
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": f"Prediction failed: {e}"})


# --- Main ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print("TomatoGuard - Step 3 of 3:  Web server")
    print("=" * 60)
    server = ThreadingHTTPServer((HOST, PORT), TomatoHandler)
    url = f"http://{HOST}:{PORT}/"
    print(f"[ok] Server running at  {url}")
    print(f"     Open it in your browser. Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[..] Stopping server. Bye!")
        server.server_close()


if __name__ == "__main__":
    main()
