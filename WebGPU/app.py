import os
import sys
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS

ONNX_FILE = "static/MulticlassCNN.onnx"

if not os.path.exists(ONNX_FILE):
    sys.exit(
        f"\n[ERROR] ONNX model not found at '{ONNX_FILE}'.\n"
        f"  Run:  python train.py\n"
        f"  Then: python app.py\n"
    )

app = Flask(__name__, template_folder='templates')
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/api/status")
def status():
    size_kb = os.path.getsize(ONNX_FILE) // 1024
    return jsonify({
        "model": "CNN (2 conv blocks + FC head)",
        "onnx": ONNX_FILE,
        "size_kb": size_kb,
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    size_kb = os.path.getsize(ONNX_FILE) // 1024
    print(f"  Model : {ONNX_FILE}  ({size_kb} KB)")
    print(f"  Open  : http://localhost:5000\n")
    app.run(debug=True, port=5000)