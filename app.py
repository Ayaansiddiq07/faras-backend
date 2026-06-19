"""
FARAS - Complete Python Backend
Deploy on Render (render.com)
Uses DeepFace + snapshot polling (no MJPEG stream)
Much more reliable over ngrok/internet connections
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import sqlite3
import os
import io
import csv
import threading
import time
import urllib.request
from datetime import datetime, date
import base64

app = Flask(__name__)
CORS(app)

# ─── Config ───────────────────────────────────────────────────────
KNOWN_FACES_DIR     = "known_faces"
DB_PATH             = "faras.db"
ADMIN_PASSWORD      = "faras2025"
SNAPSHOT_INTERVAL   = 2.0     # fetch one frame every 2 seconds
NO_FACE_RESET       = 6       # seconds of no face → reset to WAITING

# ─── State ────────────────────────────────────────────────────────
current_result      = "WAITING"
current_name        = ""
known_faces         = []
stream_thread       = None
stream_running      = False
ESP32_CAPTURE_URL   = ""      # e.g. https://xxxx.ngrok-free.app/capture
last_face_seen_time = 0

# ─── DeepFace lazy load ───────────────────────────────────────────
_deepface = None
def get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface

# ══════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, usn TEXT,
        status TEXT NOT NULL, timestamp TEXT NOT NULL,
        date TEXT NOT NULL, time TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, usn TEXT UNIQUE NOT NULL,
        image_path TEXT)""")
    conn.commit()
    conn.close()
    print("[DB] Ready.")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def is_already_logged(name):
    today = date.today().isoformat()
    conn  = get_db()
    c     = conn.cursor()
    c.execute("SELECT id FROM attendance WHERE LOWER(name)=LOWER(?) AND date=?", (name, today))
    found = c.fetchone() is not None
    conn.close()
    return found

def log_attendance(name):
    if is_already_logged(name):
        return "already_logged"
    now      = datetime.now()
    today    = date.today().isoformat()
    conn     = get_db()
    c        = conn.cursor()
    c.execute("SELECT usn FROM students WHERE LOWER(name)=LOWER(?)", (name,))
    row      = c.fetchone()
    usn      = row["usn"] if row else "N/A"
    c.execute(
        "INSERT INTO attendance (name,usn,status,timestamp,date,time) VALUES (?,?,?,?,?,?)",
        (name, usn, "Present", now.strftime("%Y-%m-%d %H:%M:%S"),
         today, now.strftime("%H:%M:%S"))
    )
    conn.commit()
    conn.close()
    print(f"[LOG] {name} Present at {now.strftime('%H:%M:%S')}")
    return "logged"

# ══════════════════════════════════════════════════════════════════
# FACE ENGINE
# ══════════════════════════════════════════════════════════════════
def load_known_faces():
    global known_faces
    known_faces = []
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            known_faces.append({"name": name, "path": path})
            print(f"[FACE] Loaded: {name}")
    print(f"[FACE] Total: {len(known_faces)}")

def fetch_snapshot(url):
    """Fetch single JPEG from ESP32-CAM /capture endpoint"""
    req = urllib.request.Request(url, headers={"User-Agent": "FARAS/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = resp.read()
    np_arr = np.frombuffer(data, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def has_face_opencv(frame_bgr):
    """Quick OpenCV check — is there any face in frame?"""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    return len(faces) > 0

def recognize_all_faces(frame_bgr):
    """
    Detect ALL faces in frame, match each against enrolled faces.
    Returns list of matched names.
    """
    DeepFace  = get_deepface()
    matched   = []
    tmp_frame = "/tmp/faras_frame.jpg"
    cv2.imwrite(tmp_frame, frame_bgr)

    try:
        faces = DeepFace.extract_faces(
            img_path=tmp_frame,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception as e:
        print(f"[FACE] Extract error: {e}")
        return matched

    for face_obj in faces:
        if face_obj.get("confidence", 0) < 0.65:
            continue

        # Crop face region
        fa  = face_obj.get("facial_area", {})
        x,y,w,h = fa.get("x",0), fa.get("y",0), fa.get("w",frame_bgr.shape[1]), fa.get("h",frame_bgr.shape[0])
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        tmp_crop = "/tmp/faras_crop.jpg"
        cv2.imwrite(tmp_crop, crop)

        # Compare against all enrolled
        for known in known_faces:
            try:
                res = DeepFace.verify(
                    img1_path=tmp_crop,
                    img2_path=known["path"],
                    model_name="Facenet",
                    detector_backend="skip",
                    enforce_detection=False,
                    distance_metric="cosine"
                )
                if res.get("verified", False):
                    name = known["name"]
                    if name not in matched:
                        matched.append(name)
                    break
            except Exception:
                continue

    return matched

# ══════════════════════════════════════════════════════════════════
# SNAPSHOT POLLING LOOP
# ══════════════════════════════════════════════════════════════════
def recognition_loop(capture_url):
    global current_result, current_name, stream_running, last_face_seen_time
    stream_running = True
    print(f"[SNAP] Polling: {capture_url}")

    while stream_running:
        try:
            frame = fetch_snapshot(capture_url)
            if frame is None:
                time.sleep(SNAPSHOT_INTERVAL)
                continue

            # Quick check — any face at all?
            if not has_face_opencv(frame):
                if time.time() - last_face_seen_time > NO_FACE_RESET:
                    current_result = "WAITING"
                    current_name   = ""
                time.sleep(SNAPSHOT_INTERVAL)
                continue

            last_face_seen_time = time.time()

            # Full recognition
            matched = recognize_all_faces(frame)

            if matched:
                name   = matched[0]
                status = log_attendance(name)
                current_result = "ALREADY_LOGGED" if status == "already_logged" else "KNOWN"
                current_name   = name
                # Log any additional faces silently
                for extra in matched[1:]:
                    log_attendance(extra)
            else:
                current_result = "UNKNOWN"
                current_name   = ""

        except urllib.error.HTTPError as e:
            print(f"[SNAP] HTTP {e.code}: {e.reason}")
            time.sleep(3)
        except Exception as e:
            print(f"[SNAP] Error: {e}")
            time.sleep(3)

        time.sleep(SNAPSHOT_INTERVAL)

# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    if data.get("password","") == ADMIN_PASSWORD:
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Wrong password"}), 401

@app.route("/api/result")
def result():
    # NodeMCU gets plain text
    if current_result in ("KNOWN", "ALREADY_LOGGED"):
        return "KNOWN"
    if current_result == "UNKNOWN":
        return "UNKNOWN"
    return "WAITING"

@app.route("/api/dashboard")
def dashboard():
    today = date.today().isoformat()
    conn  = get_db()
    c     = conn.cursor()
    c.execute("SELECT COUNT(*) as t FROM students")
    total = c.fetchone()["t"]
    c.execute("SELECT COUNT(*) as p FROM attendance WHERE date=?", (today,))
    present = c.fetchone()["p"]
    c.execute("SELECT name,usn,time,status FROM attendance WHERE date=? ORDER BY time DESC", (today,))
    records = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({
        "total_students":  total,
        "present_today":   present,
        "absent_today":    max(0, total - present),
        "attendance_rate": round(present/total*100, 1) if total else 0,
        "today_records":   records,
        "current_result":  current_result,
        "current_name":    current_name,
        "date":            today
    })

@app.route("/api/attendance")
def attendance():
    date_f = request.args.get("date","")
    name_f = request.args.get("name","")
    conn   = get_db()
    c      = conn.cursor()
    q      = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date_f: q += " AND date=?"; params.append(date_f)
    if name_f: q += " AND LOWER(name) LIKE ?"; params.append(f"%{name_f.lower()}%")
    q += " ORDER BY timestamp DESC LIMIT 200"
    c.execute(q, params)
    records = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"records": records, "count": len(records)})

@app.route("/api/report/csv")
def report_csv():
    date_f = request.args.get("date", date.today().isoformat())
    conn   = get_db()
    c      = conn.cursor()
    c.execute("SELECT name,usn,status,date,time,timestamp FROM attendance WHERE date=? ORDER BY time", (date_f,))
    records = c.fetchall()
    conn.close()
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Name","USN","Status","Date","Time","Full Timestamp"])
    for r in records:
        w.writerow([r["name"],r["usn"],r["status"],r["date"],r["time"],r["timestamp"]])
    out.seek(0)
    return send_file(io.BytesIO(out.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name=f"FARAS_{date_f}.csv")

@app.route("/api/students", methods=["GET"])
def get_students():
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT * FROM students ORDER BY name")
    s = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"students": s})

@app.route("/api/students", methods=["POST"])
def add_student():
    data = request.json or {}
    name = data.get("name","").strip()
    usn  = data.get("usn","").strip().upper()
    if not name or not usn:
        return jsonify({"success": False, "message": "Name and USN required"}), 400
    conn = get_db()
    c    = conn.cursor()
    try:
        c.execute("INSERT INTO students (name,usn) VALUES (?,?)", (name, usn))
        conn.commit(); conn.close()
        return jsonify({"success": True, "message": f"{name} added."})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "USN already exists."}), 400

@app.route("/api/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = get_db()
    c    = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.commit(); conn.close()
    return jsonify({"success": True})

@app.route("/api/enroll", methods=["POST"])
def enroll_face():
    data  = request.json or {}
    name  = data.get("name","").strip().replace(" ","_").lower()
    image = data.get("image","")
    if not name or not image:
        return jsonify({"success": False, "message": "Name and image required"}), 400
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    img_data = base64.b64decode(image.split(",")[-1])
    with open(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), "wb") as f:
        f.write(img_data)
    load_known_faces()
    return jsonify({"success": True, "message": f"Enrolled: {name}"})

@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    global stream_thread, stream_running, ESP32_CAPTURE_URL
    data = request.json or {}
    url  = data.get("url","").strip()
    if not url:
        return jsonify({"success": False, "message": "URL required"}), 400
    # Stop existing
    stream_running = False
    time.sleep(1.5)
    ESP32_CAPTURE_URL = url
    stream_thread = threading.Thread(target=recognition_loop, args=(url,), daemon=True)
    stream_thread.start()
    return jsonify({"success": True, "message": f"Started: {url}"})

@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    global stream_running
    stream_running = False
    return jsonify({"success": True})

@app.route("/api/stream/status")
def stream_status():
    return jsonify({
        "running":        stream_running,
        "current_result": current_result,
        "current_name":   current_name,
        "capture_url":    ESP32_CAPTURE_URL
    })

@app.route("/")
def health():
    return jsonify({
        "status":   "FARAS Backend Running",
        "enrolled": len(known_faces),
        "names":    [f["name"] for f in known_faces]
    })

# ══════════════════════════════════════════════════════════════════
init_db()
load_known_faces()
print("[FARAS] Ready.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
