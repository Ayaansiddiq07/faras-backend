"""
FARAS - Complete Python Backend
Deploy on Render (render.com)
Uses DeepFace with opencv backend - no dlib, no cmake

Fixes in this version:
  - Same face does not re-trigger if already logged today
  - Multiple faces in frame all get checked
  - result resets to WAITING when no face is in frame
  - "already_logged" status returned separately so UI can show it
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
KNOWN_FACES_DIR = "known_faces"
DB_PATH         = "faras.db"
ADMIN_PASSWORD  = "faras2025"        # Change this before going live
THRESHOLD       = 0.6                # DeepFace cosine distance threshold
NO_FACE_RESET   = 5                  # seconds of no face before resetting result

# ─── State ────────────────────────────────────────────────────────
current_result       = "WAITING"     # WAITING / KNOWN / UNKNOWN / ALREADY_LOGGED
current_name         = ""
known_faces          = []            # [{"name": str, "path": str}]
stream_thread        = None
stream_running       = False
ESP32_STREAM_URL     = ""
last_face_seen_time  = 0             # unix timestamp

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
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT NOT NULL,
        usn       TEXT,
        status    TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        date      TEXT NOT NULL,
        time      TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS students (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        usn        TEXT UNIQUE NOT NULL,
        image_path TEXT
    )""")
    conn.commit()
    conn.close()
    print("[DB] Initialized.")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def is_already_logged(name):
    """Check if this student was already marked present today"""
    today = date.today().isoformat()
    conn  = get_db()
    c     = conn.cursor()
    c.execute(
        "SELECT id FROM attendance WHERE LOWER(name)=LOWER(?) AND date=?",
        (name, today)
    )
    found = c.fetchone() is not None
    conn.close()
    return found

def log_attendance(name):
    """Log attendance — returns 'logged' or 'already_logged'"""
    if is_already_logged(name):
        print(f"[LOG] {name} already logged today — skip")
        return "already_logged"

    now       = datetime.now()
    today     = date.today().isoformat()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    time_str  = now.strftime("%H:%M:%S")

    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT usn FROM students WHERE LOWER(name)=LOWER(?)", (name,))
    row = c.fetchone()
    usn = row["usn"] if row else "N/A"

    c.execute(
        "INSERT INTO attendance (name,usn,status,timestamp,date,time) VALUES (?,?,?,?,?,?)",
        (name, usn, "Present", timestamp, today, time_str)
    )
    conn.commit()
    conn.close()
    print(f"[LOG] {name} marked Present at {time_str}")
    return "logged"

# ══════════════════════════════════════════════════════════════════
# FACE RECOGNITION ENGINE
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
    print(f"[FACE] Total enrolled: {len(known_faces)}")

def recognize_faces_in_frame(frame_bgr):
    """
    Check ALL faces in frame against ALL enrolled faces.
    Returns list of matched names.
    Skips anyone already logged today.
    """
    DeepFace = get_deepface()
    matched  = []

    if not known_faces:
        return matched

    tmp_path = "/tmp/faras_frame.jpg"
    cv2.imwrite(tmp_path, frame_bgr)

    try:
        # Extract all faces in frame first
        faces = DeepFace.extract_faces(
            img_path=tmp_path,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception as e:
        print(f"[FACE] Extract error: {e}")
        return matched

    if not faces:
        return matched

    for face_obj in faces:
        confidence = face_obj.get("confidence", 0)
        if confidence < 0.7:
            continue  # skip low confidence detections

        # Save cropped face temporarily
        facial_area = face_obj.get("facial_area", {})
        x = facial_area.get("x", 0)
        y = facial_area.get("y", 0)
        w = facial_area.get("w", frame_bgr.shape[1])
        h = facial_area.get("h", frame_bgr.shape[0])
        cropped = frame_bgr[y:y+h, x:x+w]
        if cropped.size == 0:
            continue
        crop_path = "/tmp/faras_crop.jpg"
        cv2.imwrite(crop_path, cropped)

        # Compare this face against all enrolled faces
        for known in known_faces:
            try:
                result = DeepFace.verify(
                    img1_path=crop_path,
                    img2_path=known["path"],
                    model_name="Facenet",
                    detector_backend="skip",   # face already cropped
                    enforce_detection=False,
                    distance_metric="cosine"
                )
                if result.get("verified", False):
                    name = known["name"]
                    if name not in matched:
                        matched.append(name)
                    break  # this face matched, move to next face
            except Exception as e:
                print(f"[FACE] Verify error for {known['name']}: {e}")
                continue

    return matched

def recognition_loop(stream_url):
    global current_result, current_name, stream_running, last_face_seen_time
    stream_running = True
    print(f"[STREAM] Starting: {stream_url}")

    while stream_running:
        try:
            stream = urllib.request.urlopen(stream_url, timeout=10)
            buffer = bytes()

            while stream_running:
                chunk  = stream.read(1024)
                buffer += chunk
                start  = buffer.find(b'\xff\xd8')
                end    = buffer.find(b'\xff\xd9')

                if start == -1 or end == -1:
                    continue

                jpg    = buffer[start:end+2]
                buffer = buffer[end+2:]

                np_arr = np.frombuffer(jpg, dtype=np.uint8)
                frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Resize to 320x240 for speed
                frame = cv2.resize(frame, (320, 240))

                # Quick OpenCV face detect to check if any face is present
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

                if len(faces) == 0:
                    # No face in frame
                    # If no face for NO_FACE_RESET seconds, reset result
                    if time.time() - last_face_seen_time > NO_FACE_RESET:
                        current_result = "WAITING"
                        current_name   = ""
                    time.sleep(0.3)
                    continue

                # Face detected — update last seen time
                last_face_seen_time = time.time()

                # Run DeepFace recognition
                matched_names = recognize_faces_in_frame(frame)

                if matched_names:
                    # Process first matched face
                    name   = matched_names[0]
                    status = log_attendance(name)

                    if status == "already_logged":
                        current_result = "ALREADY_LOGGED"
                    else:
                        current_result = "KNOWN"

                    current_name = name

                    # If multiple faces matched, log them all silently
                    for extra_name in matched_names[1:]:
                        log_attendance(extra_name)
                else:
                    current_result = "UNKNOWN"
                    current_name   = ""

                time.sleep(0.5)  # 2 FPS — saves CPU on Render free tier

        except Exception as e:
            print(f"[STREAM] Error: {e}. Retry in 3s...")
            current_result = "WAITING"
            current_name   = ""
            time.sleep(3)

# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    if data.get("password", "") == ADMIN_PASSWORD:
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Wrong password"}), 401

# NodeMCU polls this - plain text response
@app.route("/api/result")
def result():
    # NodeMCU only needs KNOWN or UNKNOWN
    # ALREADY_LOGGED = treat as KNOWN (already present, no need to re-beep)
    if current_result == "KNOWN":
        return "KNOWN"
    elif current_result == "ALREADY_LOGGED":
        return "KNOWN"
    elif current_result == "UNKNOWN":
        return "UNKNOWN"
    else:
        return "WAITING"

@app.route("/api/dashboard")
def dashboard():
    today = date.today().isoformat()
    conn  = get_db()
    c     = conn.cursor()
    c.execute("SELECT COUNT(*) as total FROM students")
    total = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as present FROM attendance WHERE date=?", (today,))
    present = c.fetchone()["present"]
    c.execute(
        "SELECT name,usn,time,status FROM attendance WHERE date=? ORDER BY time DESC",
        (today,)
    )
    records = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({
        "total_students":  total,
        "present_today":   present,
        "absent_today":    max(0, total - present),
        "attendance_rate": round(present / total * 100, 1) if total else 0,
        "today_records":   records,
        "current_result":  current_result,
        "current_name":    current_name,
        "date":            today
    })

@app.route("/api/attendance")
def attendance():
    date_f = request.args.get("date", "")
    name_f = request.args.get("name", "")
    conn   = get_db()
    c      = conn.cursor()
    query  = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date_f:
        query += " AND date=?"; params.append(date_f)
    if name_f:
        query += " AND LOWER(name) LIKE ?"; params.append(f"%{name_f.lower()}%")
    query += " ORDER BY timestamp DESC LIMIT 200"
    c.execute(query, params)
    records = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"records": records, "count": len(records)})

@app.route("/api/report/csv")
def report_csv():
    date_f = request.args.get("date", date.today().isoformat())
    conn   = get_db()
    c      = conn.cursor()
    c.execute(
        "SELECT name,usn,status,date,time,timestamp FROM attendance WHERE date=? ORDER BY time ASC",
        (date_f,)
    )
    records = c.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "USN", "Status", "Date", "Time", "Full Timestamp"])
    for r in records:
        writer.writerow([r["name"], r["usn"], r["status"], r["date"], r["time"], r["timestamp"]])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"FARAS_Attendance_{date_f}.csv"
    )

@app.route("/api/students", methods=["GET"])
def get_students():
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT * FROM students ORDER BY name")
    students = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"students": students})

@app.route("/api/students", methods=["POST"])
def add_student():
    data = request.json or {}
    name = data.get("name", "").strip()
    usn  = data.get("usn",  "").strip().upper()
    if not name or not usn:
        return jsonify({"success": False, "message": "Name and USN required"}), 400
    conn = get_db()
    c    = conn.cursor()
    try:
        c.execute("INSERT INTO students (name,usn) VALUES (?,?)", (name, usn))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"{name} added."})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "USN already exists."}), 400

@app.route("/api/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = get_db()
    c    = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/enroll", methods=["POST"])
def enroll_face():
    data  = request.json or {}
    name  = data.get("name", "").strip().replace(" ", "_").lower()
    image = data.get("image", "")
    if not name or not image:
        return jsonify({"success": False, "message": "Name and image required"}), 400
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    img_data = base64.b64decode(image.split(",")[-1])
    img_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    with open(img_path, "wb") as f:
        f.write(img_data)
    load_known_faces()
    return jsonify({"success": True, "message": f"Face enrolled for {name}"})

@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    global stream_thread, stream_running, ESP32_STREAM_URL
    data = request.json or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"success": False, "message": "Stream URL required"}), 400
    # Stop existing stream first
    stream_running = False
    time.sleep(1.5)
    ESP32_STREAM_URL = url
    stream_thread = threading.Thread(target=recognition_loop, args=(url,), daemon=True)
    stream_thread.start()
    return jsonify({"success": True, "message": f"Stream started: {url}"})

@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    global stream_running
    stream_running = False
    return jsonify({"success": True, "message": "Stream stopped."})

@app.route("/api/stream/status")
def stream_status():
    return jsonify({
        "running":        stream_running,
        "current_result": current_result,
        "current_name":   current_name,
        "stream_url":     ESP32_STREAM_URL
    })

@app.route("/")
def health():
    return jsonify({
        "status":   "FARAS Backend Running",
        "enrolled": len(known_faces),
        "names":    [f["name"] for f in known_faces]
    })

# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════
init_db()
load_known_faces()
print("[FARAS] Backend ready.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
