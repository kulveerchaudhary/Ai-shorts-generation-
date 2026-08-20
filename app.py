"""
AI Shorts Generator - Flask API
=================================
Endpoints:
  POST /api/upload            -> upload a video file, returns job_id
  POST /api/process/<job_id>  -> kicks off the pipeline in a background thread
  GET  /api/status/<job_id>   -> polling endpoint {progress, message, done}
  GET  /api/results/<job_id>  -> final manifest (list of Moments)
  GET  /media/<path>          -> serves generated clip / thumbnail files

Run with:  python3 app.py   (serves on http://localhost:5001)
"""

import os
import uuid
import threading
import traceback

from flask import Flask, request, jsonify, send_from_directory

import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB — fits free-tier RAM/disk

JOBS = {}  # job_id -> {progress, message, done, error, manifest}
JOBS_LOCK = threading.Lock()

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "no file field 'video' in request"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported extension {ext}"}), 400

    job_id = str(uuid.uuid4())[:12]
    dest = os.path.join(pipeline.UPLOAD_DIR, f"{job_id}{ext}")
    f.save(dest)

    with JOBS_LOCK:
        JOBS[job_id] = {"progress": 0, "message": "Uploaded", "done": False,
                         "error": None, "manifest": None, "video_path": dest}

    return jsonify({"job_id": job_id})


def _run_job(job_id, target_clips):
    def progress_cb(pct, msg):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["message"] = msg
    try:
        video_path = JOBS[job_id]["video_path"]
        manifest = pipeline.process_video(video_path, job_id, target_clips=target_clips,
                                           progress_cb=progress_cb)
        with JOBS_LOCK:
            JOBS[job_id]["manifest"] = manifest
            JOBS[job_id]["done"] = True
            JOBS[job_id]["progress"] = 100
    except Exception as e:
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["done"] = True


@app.route("/api/process/<job_id>", methods=["POST"])
def process(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "unknown job_id"}), 404
    target_clips = int(request.json.get("target_clips", 5)) if request.is_json else 5
    t = threading.Thread(target=_run_job, args=(job_id, target_clips), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/status/<job_id>")
def status(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "unknown job_id"}), 404
    j = JOBS[job_id]
    return jsonify({"progress": j["progress"], "message": j["message"],
                     "done": j["done"], "error": j["error"]})


@app.route("/api/results/<job_id>")
def results(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "unknown job_id"}), 404
    j = JOBS[job_id]
    if not j["done"]:
        return jsonify({"error": "not finished"}), 409
    if j["error"]:
        return jsonify({"error": j["error"]}), 500
    return jsonify(j["manifest"])


@app.route("/media/<path:relpath>")
def media(relpath):
    return send_from_directory(pipeline.OUTPUT_DIR, relpath)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
