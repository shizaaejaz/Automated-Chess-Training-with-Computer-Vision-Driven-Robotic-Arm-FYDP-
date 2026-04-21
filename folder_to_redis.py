"""
folder_to_redis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Testing feeder — reads images from a local folder and
pushes them to Redis, exactly like camera_capture_android.py.

Use this when you don't have the physical board/phone.
Swap back to camera_capture_android.py for live testing.

REDIS KEYS WRITTEN:
  latest_frame     → frame bytes (binary JPEG)
  latest_frame_id  → frame counter (0, 1, 2 ...)

HOW TO RUN:
  Terminal 1: python folder_to_redis.py
  Terminal 2: python aruco_calibration.py
  Terminal 3: python main.py  (after calibration done)

CONTROLS (while running):
  → / d   : Next image manually
  ← / a   : Previous image manually
  Space   : Pause / Resume auto-advance
  q       : Quit
  r       : Restart from first image

FOLDER STRUCTURE (just put images here):
  your_project/
  ├── images/          ← default folder (change IMAGES_FOLDER below)
  │   ├── frame_001.jpg
  │   ├── frame_002.jpg
  │   └── ...
  ├── folder_to_redis.py
  └── aruco_calibration.py
"""

import cv2
import numpy as np
import redis
import time
import os
import sys
import glob

# ── Config ────────────────────────────────────────────────────────────────────
IMAGES_FOLDER    = "images"          # folder name relative to this script
FRAME_INTERVAL   = 3.0               # seconds between auto-advance (match camera script)
LOOP_IMAGES      = True              # loop back to start when folder ends
RESIZE_TO        = (640, 480)        # resize to match camera output

REDIS_HOST       = "127.0.0.1"
REDIS_PORT       = 6379

SUPPORTED_EXT    = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# ── Collect images ────────────────────────────────────────────────────────────
def collect_images(folder):
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    files = sorted(set(files))
    return files

# ── Redis connect ─────────────────────────────────────────────────────────────
def connect_redis():
    r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r_bin.ping()
    return r_bin, r_str

# ── Push frame to Redis ───────────────────────────────────────────────────────
def push_frame(r_bin, r_str, frame, frame_count):
    resized          = cv2.resize(frame, RESIZE_TO)
    _, buffer        = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    frame_bytes      = buffer.tobytes()
    r_bin.set("latest_frame",    frame_bytes)
    r_str.set("latest_frame_id", str(frame_count))
    return len(frame_bytes)

# ── Draw HUD on preview ───────────────────────────────────────────────────────
def draw_hud(preview, idx, total, filename, frame_count, paused, next_in):
    h, w = preview.shape[:2]
    overlay = preview.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 30), -1)
    status = "  ⏸ PAUSED" if paused else f"  ▶ Next in {next_in:.1f}s"
    cv2.putText(overlay, f"[{idx+1}/{total}] {os.path.basename(filename)}{status}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)

    # Bottom bar
    cv2.rectangle(overlay, (0, h - 34), (w, h), (20, 20, 30), -1)
    cv2.putText(overlay,
                "A/← Prev  |  D/→ Next  |  Space Pause  |  R Restart  |  Q Quit",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 200), 1)

    # Redis badge
    badge = f"Redis frame_id: {frame_count}"
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(overlay, (w - bw - 14, 4), (w - 4, 4 + bh + 8), (0, 160, 80), -1)
    cv2.putText(overlay, badge, (w - bw - 10, 4 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.88, preview, 0.12, 0, preview)
    return preview

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  📂  Folder → Redis Feeder (Testing Mode)")
    print("=" * 55)

    # Validate folder
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGES_FOLDER)
    if not os.path.isdir(folder):
        print(f"\n❌  Folder not found: {folder}")
        print(f"    Create it and put your test images inside:")
        print(f"    mkdir {IMAGES_FOLDER}")
        sys.exit(1)

    images = collect_images(folder)
    if not images:
        print(f"\n❌  No images found in: {folder}")
        print(f"    Supported: jpg, jpeg, png, bmp, webp")
        sys.exit(1)

    print(f"\n📁 Folder  : {folder}")
    print(f"🖼️  Images  : {len(images)} found")
    print(f"⏱️  Interval: {FRAME_INTERVAL} sec")
    print(f"🔁 Loop    : {LOOP_IMAGES}")
    print()

    for i, f in enumerate(images):
        print(f"   [{i+1:03d}] {os.path.basename(f)}")
    print()

    # Redis
    print("🔌 Connecting to Redis...")
    try:
        r_bin, r_str = connect_redis()
        print("✅ Redis connected!\n")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Run: redis-server")
        sys.exit(1)

    cv2.namedWindow("Folder → Redis  [Test Feeder]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Folder → Redis  [Test Feeder]", 720, 560)

    idx          = 0
    frame_count  = 0
    paused       = False
    last_push    = time.time() - FRAME_INTERVAL  # push immediately on start
    next_in      = 0.0

    print("📸 Running — controls: A/D = prev/next  |  Space = pause  |  Q = quit\n")

    while True:
        filename = images[idx]
        frame    = cv2.imread(filename)

        if frame is None:
            print(f"⚠️  Could not read: {filename} — skipping")
            idx = (idx + 1) % len(images)
            continue

        now     = time.time()
        elapsed = now - last_push
        next_in = max(0.0, FRAME_INTERVAL - elapsed)

        # Auto-push on interval (or if first frame)
        if not paused and elapsed >= FRAME_INTERVAL:
            size = push_frame(r_bin, r_str, frame, frame_count)
            print(f"✅ Redis ← [{idx+1}/{len(images)}] "
                  f"{os.path.basename(filename):30s} | "
                  f"frame_id: {frame_count:04d} | "
                  f"{size//1024} KB | "
                  f"{time.strftime('%H:%M:%S')}")
            frame_count += 1
            last_push    = now
            next_in      = FRAME_INTERVAL

            # Auto-advance to next image
            if not paused:
                idx += 1
                if idx >= len(images):
                    if LOOP_IMAGES:
                        idx = 0
                        print("\n🔁 Loop — starting from first image\n")
                    else:
                        print("\n✅ All images pushed — done!")
                        break

        # Preview
        preview = cv2.resize(frame, (680, 500))
        preview = draw_hud(preview, idx, len(images), filename,
                           frame_count, paused, next_in)
        cv2.imshow("Folder → Redis  [Test Feeder]", preview)

        # Key handling
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            print("\n🛑 Quit.")
            break

        elif key == ord(' '):
            paused = not paused
            print(f"{'⏸  Paused' if paused else '▶  Resumed'}")

        elif key in (ord('d'), 83):   # d or →
            idx = (idx + 1) % len(images)
            # Force immediate push of this frame
            size = push_frame(r_bin, r_str, cv2.imread(images[idx]), frame_count)
            print(f"→ Manual next → [{idx+1}/{len(images)}] "
                  f"{os.path.basename(images[idx])} | frame_id: {frame_count}")
            frame_count += 1
            last_push    = time.time()

        elif key in (ord('a'), 81):   # a or ←
            idx = (idx - 1) % len(images)
            size = push_frame(r_bin, r_str, cv2.imread(images[idx]), frame_count)
            print(f"← Manual prev → [{idx+1}/{len(images)}] "
                  f"{os.path.basename(images[idx])} | frame_id: {frame_count}")
            frame_count += 1
            last_push    = time.time()

        elif key == ord('r'):
            idx         = 0
            frame_count = 0
            last_push   = time.time() - FRAME_INTERVAL
            print("🔄 Restart — back to first image")

    cv2.destroyAllWindows()
    print(f"\n✅ Done — {frame_count} frames pushed to Redis total")


if __name__ == "__main__":
    main()