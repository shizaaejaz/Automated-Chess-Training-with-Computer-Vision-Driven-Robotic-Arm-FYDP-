"""
camera_capture_android.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Android IP Webcam se frames leta hai (WiFi).
Frames ab disk pe NAHI — seedha Redis mein store hote hain.

NOTE: Frames are rotated 180 degrees to compensate for tripod orientation.
"""

import cv2
import time
import urllib.request
import numpy as np
import redis

# ── Config ────────────────────────────────────────────────
PHONE_IP         = "10.200.253.238"  # Adjusted if typo was present
PORT             = 8080
CAPTURE_INTERVAL = 3          # har kitne second baad frame lo

REDIS_HOST       = "127.0.0.1"
REDIS_PORT       = 6379

SHOT_URL = f"http://{PHONE_IP}:{PORT}/shot.jpg"

# ── Redis connections ─────────────────────────────────────
r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ── Startup ───────────────────────────────────────────────
print("=" * 50)
print("  📱 Android Camera Capture Started (180° Rotated)")
print(f"  📡 URL      : {SHOT_URL}")
print(f"  🗄️  Redis    : {REDIS_HOST}:{REDIS_PORT}")
print(f"  ⏱️  Interval : {CAPTURE_INTERVAL} sec")
print("=" * 50)

# ── Connection test ───────────────────────────────────────
print("\n🔍 Phone se connection test kar raha hoon...")
try:
    urllib.request.urlopen(SHOT_URL, timeout=5)
    print("✅ Phone connected!\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# ── Redis test ────────────────────────────────────────────
try:
    r_bin.ping()
    print("✅ Redis connected!\n")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    exit(1)

# ── Main loop ─────────────────────────────────────────────
frame_count  = 0
last_capture = time.time() - CAPTURE_INTERVAL

print("📸 Capturing frames — Ctrl+C ya 'q' se band karo\n")

while True:
    try:
        now = time.time()

        # ── Capture interval check ────────────────────────
        if now - last_capture >= CAPTURE_INTERVAL:

            # Step 1: Phone se photo lo
            resp   = urllib.request.urlopen(SHOT_URL, timeout=5)
            img_np = np.frombuffer(resp.read(), dtype=np.uint8)
            frame  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠️  Frame decode failed — retry...")
                time.sleep(2)
                continue

            # ROTATION: 90 degrees Counter-Clockwise (White at Bottom fix)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Step 2: Resize
            frame = cv2.resize(frame, (640, 480))

            # Step 3: JPEG bytes mein encode karo
            _, buffer    = cv2.imencode('.jpg', frame)
            frame_bytes  = buffer.tobytes()

            # Step 4: Redis mein store karo
            r_bin.set("latest_frame",    frame_bytes)
            r_str.set("latest_frame_id", str(frame_count))

            size_kb = len(frame_bytes) // 1024
            print(f"✅ Redis → frame_id: {frame_count:04d} | size: {size_kb} KB | time: {time.strftime('%H:%M:%S')}")

            frame_count  += 1
            last_capture  = now

        # ── Live preview window ───────────────────────────
        try:
            resp2   = urllib.request.urlopen(SHOT_URL, timeout=3)
            img_np2 = np.frombuffer(resp2.read(), dtype=np.uint8)
            preview = cv2.imdecode(img_np2, cv2.IMREAD_COLOR)

            if preview is not None:
                # ROTATION for preview
                preview = cv2.rotate(preview, cv2.ROTATE_90_COUNTERCLOCKWISE)
                preview   = cv2.resize(preview, (480, 360))
                remaining = max(0, int(CAPTURE_INTERVAL - (time.time() - last_capture)))

                cv2.putText(
                    preview,
                    f"Frame: {frame_count} | Next: {remaining}s | Redis: OK",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )
                cv2.imshow("Android Camera — Chess Robot", preview)

        except Exception:
            pass

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"⚠️  Error: {e}")
        time.sleep(2)

cv2.destroyAllWindows()
print(f"\n✅ Done — Total frames captured: {frame_count}")
