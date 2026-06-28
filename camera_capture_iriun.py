"""
camera_capture_iriun.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iPhone + Iriun Webcam se raw (full-resolution) frames lekar Redis mein feed karta hai.
Zero downscaling, zero filtering, and maximum JPEG quality for YOLO and warping.

Requirements:
  1. Iriun Webcam app running on iPhone.
  2. iPhone connected to PC via USB cable (turn off iPhone Wi-Fi to force USB connection).
  3. Iriun Webcam driver installed and running on PC.
  4. Redis server running locally.
"""

import cv2
import time
import redis
import numpy as np

# ── Config ────────────────────────────────────────────────
CAPTURE_INTERVAL = 3               # Har kitne second baad frame Redis mein push karein
REDIS_HOST       = "127.0.0.1"
REDIS_PORT       = 6379
CAMERA_INDEX     = 1    # Set to 0, 1, 2, etc. to force a specific camera index. If None, auto-picks.

# Rotation Code:
#   None                            -> No rotation (Landscape)
#   cv2.ROTATE_90_COUNTERCLOCKWISE  -> 90 degrees CCW (matches camera_capture_android.py)
#   cv2.ROTATE_90_CLOCKWISE         -> 90 degrees CW
#   cv2.ROTATE_180                  -> 180 degrees
ROTATION_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE 

# ── Redis Connection ──────────────────────────────────────
r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ── Camera Setup (Search for active Iriun Webcam) ──────────
cap = None
chosen_index = None

print("=" * 60)
print("  📸 Iriun Webcam Raw Capture Starter")
print(f"  🗄️  Redis    : {REDIS_HOST}:{REDIS_PORT}")
print(f"  ⏱️  Interval : {CAPTURE_INTERVAL} sec")
print("=" * 60)

# Connect to Redis
try:
    r_bin.ping()
    print("✅ Redis connected!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    exit(1)

# Find all active camera devices using pygrabber
print("\n🔍 Scanning for camera devices on your PC...")
try:
    from pygrabber.dshow_graph import FilterGraph
    devices = FilterGraph().get_input_devices()
except Exception:
    devices = []

if not devices:
    print("\n❌ CRITICAL ERROR: No camera devices found on system (or pygrabber failed).")
    print("💡 Please make sure your camera is connected and drivers are installed.")
    exit(1)

print("\n📸 Detected Cameras:")
for idx, name in enumerate(devices):
    print(f"   👉 Index {idx}: {name}")

# Auto-find Iriun Webcam by name
iriun_indices = [idx for idx, name in enumerate(devices) if "iriun" in name.lower()]

# Determine which index to open
if CAMERA_INDEX is not None:
    chosen_index = CAMERA_INDEX
    print(f"\n👉 Using user-configured CAMERA_INDEX: {chosen_index} ({devices[chosen_index] if chosen_index < len(devices) else 'Unknown'})")
elif iriun_indices:
    chosen_index = iriun_indices[0]
    print(f"\n✨ Auto-detected 'Iriun Webcam' at index {chosen_index}!")
else:
    # Skip index 0 (built-in camera) if other external cameras are connected
    if len(devices) > 1:
        chosen_index = 1
        print(f"\n👉 Iriun Webcam not found by name. Selecting index {chosen_index} to skip built-in laptop camera.")
    else:
        chosen_index = 0
        print(f"\n👉 Selecting default index {chosen_index}.")

cap = cv2.VideoCapture(chosen_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"\n❌ Failed to open camera at index {chosen_index}.")
    exit(1)

# Use the camera's native resolution from the Iriun Webcam PC client to avoid aspect ratio stretching.
# (If you need to force a specific resolution, make sure it matches your Iriun app's orientation)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Read resolution back to confirm
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Camera configured to: {actual_w}x{actual_h}")

frame_count = 0
last_capture = time.time() - CAPTURE_INTERVAL

print("\n📸 Capturing frames — Focus preview window and press 'q' to stop\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠️ Camera frame capture failed! Retrying...")
        time.sleep(0.5)
        continue

    now = time.time()

    # Capture interval par Redis mein feed push karein
    if now - last_capture >= CAPTURE_INTERVAL:
        
        # Apply rotation if configured
        processed_frame = frame.copy()
        if ROTATION_CODE is not None:
            processed_frame = cv2.rotate(processed_frame, ROTATION_CODE)
            
        # Encode raw frame to JPEG with 100% quality (No resizing, no filtering)
        success, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        if success:
            frame_bytes = buffer.tobytes()
            
            # Push raw bytes and ID to Redis
            r_bin.set("latest_frame", frame_bytes)
            r_str.set("latest_frame_id", str(frame_count))
            
            size_kb = len(frame_bytes) // 1024
            print(f"✅ Redis ← frame_id: {frame_count:04d} | Size: {size_kb} KB | "
                  f"Res: {processed_frame.shape[1]}x{processed_frame.shape[0]} | "
                  f"Time: {time.strftime('%H:%M:%S')}")
            
            frame_count += 1
            last_capture = now
        else:
            print("⚠️ Failed to encode frame to JPEG!")

    # Live Preview window (Resized only for screen display, Redis feed remains full raw quality)
    preview = cv2.resize(frame, (640, 480))
    
    # Overlay info on preview
    remaining = max(0, int(CAPTURE_INTERVAL - (time.time() - last_capture)))
    cv2.putText(
        preview,
        f"Iriun Index {chosen_index} | Next: {remaining}s | Press 'q' to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )
    cv2.imshow("Iriun Raw Feed Preview", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Finished — Total frames captured: {frame_count}")
