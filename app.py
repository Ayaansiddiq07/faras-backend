"""
FARAS - Complete Python Backend
Deploy on Render (render.com)

Install dependencies:
    pip install flask flask-cors face-recognition opencv-python-headless numpy requests

Folder structure:
    faras_backend/
    ├── app.py              ← this file
    ├── requirements.txt
    ├── known_faces/        ← add student face images here (name.jpg)
    └── faras.db            ← auto-created SQLite database
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import face_recognition
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
CORS(app)  # Allow Lovable frontend to call this API

# ─── Config ───────────────────────────────────────────────────────
KNOWN_FACES_DIR  = "known_faces"
DB_PATH          = "faras.db"
TOLERANCE        = 0.5
ADMIN_PASSWORD   = "faras2025"          # Change this
ESP32_STREAM_URL = ""                   # Set via /api/config endpoint

# How many consecutive "no face visible" frames must pass before we revert
# current_result back to WAITING. At ~5-10 decoded frames/sec this is
# roughly 1-2 seconds of grace - long enough to ignore a blink or brief
# head-turn, short enough that the status doesn't stay stuck once someone
# actually walks away.
NO_FACE_GRACE_FRAMES = 10

# ─── State (thread-safe access via lock) ──────────────────────────
state_lock       = threading.Lock()
current_result   = "WAITING"
current_name     = ""
known_encodings  = []
known_names      = []
stream_thread    = None
stream_running   = False

# Handle to the currently-open urllib connection to the camera. Needed so
# /api/stream/start can force-close the *socket* of a previous run, not just
# flip stream_running to False - the old thread can be blocked inside a
# blocking stream.read() call, in which case stream_thread.join(timeout=3)
# silently times out and returns while the old thread (and its open
# connection to the camera) is still alive. Most ESP32-CAM firmware only
# accepts one streaming client at a time, so two live connections to the
# same camera is what produces the 503/404 burst seen from the camera.
current_stream_handle = None

# Max buffer size to prevent memory leaks in stream parsing (5MB)
MAX_BUFFER_SIZE  = 5 * 1024 * 1024

# ══════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        c = conn.cursor()

        # Attendance records
        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                usn         TEXT,
                status      TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL
            )
        """)

        # Students enrolled
        c.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                usn         TEXT UNIQUE NOT NULL,
                image_path  TEXT
            )
        """)

        # Config table
        c.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        print("[DB] Database initialized.")
    finally:
        conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

# ══════════════════════════════════════════════════════════════════
# FACE RECOGNITION ENGINE
# ══════════════════════════════════════════════════════════════════

def load_known_faces():
    global known_encodings, known_names
    new_encodings = []
    new_names     = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        known_encodings = new_encodings
        known_names = new_names
        return

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path  = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                encs  = face_recognition.face_encodings(image)
                if encs:
                    new_encodings.append(encs[0])
                    name = os.path.splitext(filename)[0].replace("_", " ").title()
                    new_names.append(name)
                    print(f"[FACE] Loaded: {name}")
            except Exception as e:
                print(f"[FACE] Error loading {filename}: {e}")

    # Atomic swap — thread-safe
    known_encodings = new_encodings
    known_names = new_names
    print(f"[FACE] Total enrolled: {len(known_names)}")

def log_attendance(name, status):
    """Log attendance to DB — avoid duplicate in same day"""
    now       = datetime.now()
    today     = date.today().isoformat()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    time_str  = now.strftime("%H:%M:%S")

    conn = get_db()
    try:
        c = conn.cursor()

        # Get USN from students table
        c.execute("SELECT usn FROM students WHERE LOWER(name) = LOWER(?)", (name,))
        row = c.fetchone()
        usn = row["usn"] if row else "N/A"

        # Check if already logged today
        c.execute(
            "SELECT id FROM attendance WHERE LOWER(name) = LOWER(?) AND date = ?",
            (name, today)
        )
        existing = c.fetchone()

        if not existing:
            c.execute(
                "INSERT INTO attendance (name, usn, status, timestamp, date, time) VALUES (?,?,?,?,?,?)",
                (name, usn, status, timestamp, today, time_str)
            )
            conn.commit()
            print(f"[LOG] Attendance logged: {name} ({status}) at {time_str}")
        else:
            print(f"[LOG] {name} already logged today.")
    except Exception as e:
        print(f"[LOG] Error logging attendance for {name}: {e}")
    finally:
        conn.close()

def recognition_loop(stream_url):
    global current_result, current_name, stream_running, stream_thread, current_stream_handle

    print(f"[STREAM] Connecting to {stream_url}")
    stream_running = True

    while stream_running and threading.current_thread() == stream_thread:
        stream = None
        try:
            stream = urllib.request.urlopen(stream_url, timeout=10)
            with state_lock:
                current_stream_handle = stream
            buffer = bytes()

            frame_count = 0
            decode_fail_count = 0
            no_face_count = 0

            while stream_running and threading.current_thread() == stream_thread:
                chunk = stream.read(1024)
                if not chunk:
                    break  # Stream ended

                buffer += chunk

                # Prevent buffer from growing unbounded (memory leak fix)
                if len(buffer) > MAX_BUFFER_SIZE:
                    buffer = buffer[-MAX_BUFFER_SIZE:]

                # Extract ALL complete frames currently sitting in the buffer,
                # not just one - otherwise a backlog can build up and we always
                # process a stale, possibly truncated, frame.
                while True:
                    start = buffer.find(b'\xff\xd8')
                    if start == -1:
                        break
                    # IMPORTANT: search for the end marker only AFTER the start
                    # marker. \xff\xd9 byte pairs can appear inside the actual
                    # compressed image data too, not just as the real frame
                    # boundary - searching from position 0 (the old bug) can
                    # match one of those false positives and slice out a
                    # corrupted/truncated frame, which is what was causing
                    # "Corrupt JPEG data: premature end of data segment".
                    end = buffer.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        break  # End marker not arrived yet, wait for more chunks

                    jpg    = buffer[start:end + 2]
                    buffer = buffer[end + 2:]

                    # Sanity-check size before decoding - a real frame at even
                    # modest resolution is well over a few KB. Tiny "frames"
                    # are almost always a parsing artifact.
                    if len(jpg) < 1000:
                        continue

                    frame_count += 1

                    np_arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        decode_fail_count += 1
                        if decode_fail_count % 20 == 1:
                            print(f"[STREAM] Decode failed ({decode_fail_count} total, "
                                  f"{frame_count} frames seen, jpg size={len(jpg)})")
                        continue

                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    locations = face_recognition.face_locations(rgb)
                    encodings = face_recognition.face_encodings(rgb, locations)

                    if not encodings:
                        no_face_count += 1
                        if no_face_count % 20 == 1:
                            print(f"[STREAM] No face in frame ({no_face_count} total, "
                                  f"{frame_count} frames decoded OK)")

                        # BUGFIX: previously this branch just `continue`d,
                        # which meant current_result was NEVER updated when
                        # no face was visible - it stayed stuck on whatever
                        # the last detection was (e.g. "KNOWN") forever,
                        # even minutes after the person walked away.
                        #
                        # Use a short grace period (based on elapsed frames,
                        # not just a single missed frame) before reverting
                        # to WAITING, so a brief blink/head-turn doesn't
                        # cause the NodeMCU's LED/buzzer to flicker on every
                        # single no-face frame.
                        if no_face_count >= NO_FACE_GRACE_FRAMES:
                            with state_lock:
                                if current_result != "WAITING":
                                    current_result = "WAITING"
                                    current_name   = ""
                        continue

                    # A face WAS found this frame - reset the no-face streak
                    no_face_count = 0

                    print(f"[STREAM] Face detected on frame #{frame_count}")

                    detected_name = "Unknown"
                    detected      = "UNKNOWN"

                    # Take snapshot of known faces to avoid race conditions
                    local_encodings = known_encodings
                    local_names     = known_names

                    for encoding in encodings:
                        if not local_encodings:
                            break
                        matches   = face_recognition.compare_faces(local_encodings, encoding, TOLERANCE)
                        distances = face_recognition.face_distance(local_encodings, encoding)
                        best_idx  = int(np.argmin(distances))

                        if matches[best_idx]:
                            detected_name = local_names[best_idx]
                            detected      = "KNOWN"
                            log_attendance(detected_name, "Present")
                            break

                    with state_lock:
                        current_result = detected
                        current_name   = detected_name

        except Exception as e:
            print(f"[ERROR] Stream error: {e}. Retry in 3s...")
            with state_lock:
                current_result = "WAITING"
            time.sleep(3)
        finally:
            # Close stream to prevent resource leaks
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass
            with state_lock:
                if current_stream_handle is stream:
                    current_stream_handle = None

# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════

# ── Auth ──────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400
    password = data.get("password", "")
    if password == ADMIN_PASSWORD:
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Wrong password"}), 401

# ── Live Result (NodeMCU polls this) ─────────────────────────────
@app.route("/api/result")
def result():
    return current_result  # Plain text: KNOWN or UNKNOWN

# ── Dashboard Metrics ─────────────────────────────────────────────
@app.route("/api/dashboard")
def dashboard():
    today = date.today().isoformat()
    conn  = get_db()
    try:
        c = conn.cursor()

        c.execute("SELECT COUNT(*) as total FROM students")
        total_students = c.fetchone()["total"]

        c.execute("SELECT COUNT(*) as present FROM attendance WHERE date = ?", (today,))
        present_today = c.fetchone()["present"]

        absent_today = max(0, total_students - present_today)

        c.execute("""
            SELECT name, usn, time, status FROM attendance
            WHERE date = ? ORDER BY time DESC
        """, (today,))
        today_records = [dict(r) for r in c.fetchall()]
    finally:
        conn.close()

    return jsonify({
        "total_students":  total_students,
        "present_today":   present_today,
        "absent_today":    absent_today,
        "attendance_rate": round((present_today / total_students * 100), 1) if total_students else 0,
        "today_records":   today_records,
        "current_result":  current_result,
        "current_name":    current_name,
        "date":            today
    })

# ── Attendance History ────────────────────────────────────────────
@app.route("/api/attendance")
def attendance():
    date_filter = request.args.get("date", "")
    name_filter = request.args.get("name", "")

    conn  = get_db()
    try:
        c     = conn.cursor()
        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if date_filter:
            query += " AND date = ?"
            params.append(date_filter)
        if name_filter:
            query += " AND LOWER(name) LIKE ?"
            params.append(f"%{name_filter.lower()}%")

        query += " ORDER BY timestamp DESC LIMIT 200"
        c.execute(query, params)
        records = [dict(r) for r in c.fetchall()]
    finally:
        conn.close()

    return jsonify({"records": records, "count": len(records)})

# ── Download Report as CSV ────────────────────────────────────────
@app.route("/api/report/csv")
def report_csv():
    date_filter = request.args.get("date", date.today().isoformat())
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT name, usn, status, date, time, timestamp
            FROM attendance WHERE date = ?
            ORDER BY time ASC
        """, (date_filter,))
        records = c.fetchall()
    finally:
        conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "USN", "Status", "Date", "Time", "Full Timestamp"])
    for r in records:
        writer.writerow([r["name"], r["usn"], r["status"], r["date"], r["time"], r["timestamp"]])

    output.seek(0)
    byte_output = io.BytesIO(output.getvalue().encode())
    filename = f"FARAS_Attendance_{date_filter}.csv"
    return send_file(
        byte_output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

# ── Students ──────────────────────────────────────────────────────
@app.route("/api/students", methods=["GET"])
def get_students():
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM students ORDER BY name")
        students = [dict(r) for r in c.fetchall()]
    finally:
        conn.close()
    return jsonify({"students": students})

@app.route("/api/students", methods=["POST"])
def add_student():
    data = request.json
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400
    name = data.get("name", "").strip()
    usn  = data.get("usn",  "").strip().upper()

    if not name or not usn:
        return jsonify({"success": False, "message": "Name and USN required"}), 400

    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO students (name, usn) VALUES (?, ?)", (name, usn))
        conn.commit()
        return jsonify({"success": True, "message": f"{name} added."})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "USN already exists."}), 400
    finally:
        conn.close()

@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
    finally:
        conn.close()
    return jsonify({"success": True})

# ── Upload Face Image ─────────────────────────────────────────────
@app.route("/api/enroll", methods=["POST"])
def enroll_face():
    data = request.json
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400
    name  = data.get("name", "").strip().replace(" ", "_").lower()
    image = data.get("image", "")  # base64 string

    if not name or not image:
        return jsonify({"success": False, "message": "Name and image required"}), 400

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    # Decode and save image
    try:
        img_data = base64.b64decode(image.split(",")[-1])
    except Exception:
        return jsonify({"success": False, "message": "Invalid image data"}), 400

    img_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    with open(img_path, "wb") as f:
        f.write(img_data)

    # Reload face encodings
    load_known_faces()
    return jsonify({"success": True, "message": f"Face enrolled for {name}"})

# ── Start/Stop Stream ─────────────────────────────────────────────
@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    global stream_thread, stream_running, ESP32_STREAM_URL, current_stream_handle
    data = request.json
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400
    url  = data.get("url", "").strip()

    if not url:
        return jsonify({"success": False, "message": "Stream URL required"}), 400

    # Stop existing stream thread
    ESP32_STREAM_URL = url
    stream_running   = False

    # Force-close the previous connection's socket. If the old thread is
    # blocked inside stream.read(), join(timeout=3) below can time out and
    # return while that thread - and its open connection to the camera -
    # is still alive. Closing the underlying socket directly unblocks the
    # read() immediately (it raises, the thread's except/finally handles
    # it and exits cleanly) instead of leaving two simultaneous clients
    # connected to a camera that may only support one.
    with state_lock:
        old_handle = current_stream_handle
        current_stream_handle = None
    if old_handle:
        try:
            old_handle.close()
        except Exception:
            pass

    if stream_thread and stream_thread.is_alive():
        stream_thread.join(timeout=3)
        if stream_thread.is_alive():
            print("[STREAM] Warning: previous stream thread did not exit in time.")

    stream_thread = threading.Thread(
        target=recognition_loop, args=(url,), daemon=True
    )
    stream_thread.start()
    return jsonify({"success": True, "message": f"Stream started: {url}"})

@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    global stream_running, current_stream_handle
    stream_running = False
    with state_lock:
        old_handle = current_stream_handle
        current_stream_handle = None
    if old_handle:
        try:
            old_handle.close()
        except Exception:
            pass
    return jsonify({"success": True, "message": "Stream stopped."})

@app.route("/api/stream/status")
def stream_status():
    return jsonify({
        "running":        stream_running,
        "current_result": current_result,
        "current_name":   current_name,
        "stream_url":     ESP32_STREAM_URL
    })

# ── Health Check ──────────────────────────────────────────────────
@app.route("/")
def health():
    return jsonify({
        "status":  "FARAS Backend Running",
        "enrolled": len(known_names),
        "names":   known_names
    })

# ══════════════════════════════════════════════════════════════════
# INIT (runs on import — needed for gunicorn)
# ══════════════════════════════════════════════════════════════════
init_db()
load_known_faces()
print("\n[FARAS] Backend ready.")

# ══════════════════════════════════════════════════════════════════
# MAIN (local dev only)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[FARAS] Admin password:", ADMIN_PASSWORD)
    print("[FARAS] Endpoints:")
    print("  POST /api/login")
    print("  GET  /api/dashboard")
    print("  GET  /api/attendance")
    print("  GET  /api/report/csv")
    print("  GET  /api/students")
    print("  POST /api/students")
    print("  POST /api/enroll")
    print("  POST /api/stream/start")
    print("  GET  /api/result      ← NodeMCU polls this\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
