"""
aruco_calibration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess Robot — Board Calibration Phase

WHAT THIS DOES:
  1. Reads live frames from Redis (put there by folder_to_redis.py / camera)
  2. Detects ArUco markers with IDs: 10 (TL), 11 (TR), 12 (BR), 13 (BL)
  3. Shows a live annotated visualization with per-marker lock status
  4. ACCUMULATES markers across frames — each marker locks independently:
       • Marker seen MIN_SEEN_TO_LOCK times → LOCKED (position averaged)
       • All 4 locked → auto-save board_cache.json
     This means: ID10 from frame 1, ID11 from frame 5 → both count!
  5. Asks you to run main.py in the next terminal

REDIS KEYS READ:
  latest_frame     → JPEG bytes of current frame
  latest_frame_id  → frame counter (used to detect new frames)

OUTPUT FILE:
  board_cache.json → cached corner points + perspective matrix

HOW TO RUN:
  Terminal 1: python camera_capture_android.py
  Terminal 2: python aruco_calibration.py
  Terminal 3: python main.py  (after calibration succeeds)
"""

import cv2
import numpy as np
import redis
import json
import time
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST     = "127.0.0.1"
REDIS_PORT     = 6379
CACHE_FILE     = "board_cache.json"

# Marker IDs and their board positions (from white's perspective, white=bottom)
# ID  → (label, board_corner_index)
#   corner index for perspective warp:
#   0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left  (of warped output)
MARKER_CONFIG = {
    10: {"label": "TL", "corner_idx": 0, "color_bgr": (0, 255, 0)},    # green
    11: {"label": "TR", "corner_idx": 1, "color_bgr": (0, 200, 255)},  # yellow
    12: {"label": "BR", "corner_idx": 2, "color_bgr": (255, 100, 0)},  # orange
    13: {"label": "BL", "corner_idx": 3, "color_bgr": (255, 0, 200)},  # pink
}

MIN_SEEN_TO_LOCK = 3     # how many times a marker must be seen before it's "locked"
                         # lower = faster but less stable; 3 is a good balance
WARP_SIZE        = 640   # output square board image size (pixels)
DISPLAY_WIDTH    = 900   # total display window width
DISPLAY_HEIGHT   = 600   # total display window height

# ── Robust ArUco Detector Setup ───────────────────────────────────────────────
# Try multiple dictionaries — use whichever one detects your markers
ARUCO_DICTS_TO_TRY = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_6X6_250,
]

def make_detector(aruco_dict_id, adaptive=False):
    """Create an ArUco detector with optional relaxed parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()

    if adaptive:
        # Relaxed params for tricky lighting / blur
        params.adaptiveThreshWinSizeMin    = 3
        params.adaptiveThreshWinSizeMax    = 53
        params.adaptiveThreshWinSizeStep   = 4
        params.adaptiveThreshConstant      = 7
        params.minMarkerPerimeterRate      = 0.02   # detect smaller markers
        params.maxMarkerPerimeterRate      = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy  = 0.01
        params.minDistanceToBorder          = 1
        params.errorCorrectionRate          = 0.9

    return cv2.aruco.ArucoDetector(aruco_dict, params)

# Build a list of detectors to try in order (strict → relaxed)
DETECTORS = []
for d_id in ARUCO_DICTS_TO_TRY:
    DETECTORS.append(make_detector(d_id, adaptive=False))
    DETECTORS.append(make_detector(d_id, adaptive=True))

# ── Image Pre-processing ──────────────────────────────────────────────────────
def preprocess_for_aruco(frame):
    """
    Returns a list of candidate images to try ArUco detection on.
    Tries: original gray, CLAHE enhanced, sharpened, high contrast
    More candidates = better chance of detection in poor lighting.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = [gray]

    # CLAHE — adaptive contrast enhancement (best for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    candidates.append(clahe.apply(gray))

    # Sharpened — helps with slight blur from phone camera
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    candidates.append(sharpened)

    # High contrast stretch
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray.astype(float) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    candidates.append(stretched)

    # Bilateral filter — removes noise while keeping edges (good for blur)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    candidates.append(bilateral)

    return candidates

# ── Core Detection ────────────────────────────────────────────────────────────
def detect_aruco_markers(frame):
    """
    Tries every combination of preprocessed image + detector.
    Returns dict: {marker_id: center_point (x, y)} for detected markers.
    Also returns corners dict for precise warp.
    """
    best_found = {}       # id → center
    best_corners_raw = {} # id → 4 corner points (for warp)

    candidates = preprocess_for_aruco(frame)
    target_ids = set(MARKER_CONFIG.keys())

    for img_candidate in candidates:
        for detector in DETECTORS:
            corners, ids, _ = detector.detectMarkers(img_candidate)

            if ids is None:
                continue

            found = {}
            corners_raw = {}
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in target_ids:
                    pts = corners[i][0]  # shape (4, 2)
                    center = pts.mean(axis=0)
                    found[int(marker_id)] = center
                    corners_raw[int(marker_id)] = pts

            # Keep the result that found the most markers
            if len(found) > len(best_found):
                best_found = found
                best_corners_raw = corners_raw

            if len(best_found) == 4:
                return best_found, best_corners_raw  # perfect — stop early

    return best_found, best_corners_raw

# ── Perspective Warp ──────────────────────────────────────────────────────────
def compute_warp_matrix(marker_centers):
    """
    Given 4 detected marker centers (id→(x,y)), compute the perspective
    transform matrix so that the board is a top-down square with white at bottom.

    Board layout (white at bottom, white's POV):
      TL(10) ──── TR(11)
        |              |
        |              |
      BL(13) ──── BR(12)
    """
    W = WARP_SIZE
    src_pts = np.float32([
        marker_centers[10],  # TL in board space
        marker_centers[11],  # TR
        marker_centers[12],  # BR
        marker_centers[13],  # BL
    ])
    dst_pts = np.float32([
        [0,   0  ],  # TL → top-left of output
        [W-1, 0  ],  # TR → top-right
        [W-1, W-1],  # BR → bottom-right
        [0,   W-1],  # BL → bottom-left
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, src_pts, dst_pts

# ── Visualization ─────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
STATUS_W   = 280   # right-side status panel width

def draw_marker_on_frame(frame, marker_id, corners_pts, center, locked=False):
    """Draw a labeled bounding box around a detected marker."""
    cfg   = MARKER_CONFIG[marker_id]
    color = cfg["color_bgr"]
    label = f"ID{marker_id} ({cfg['label']})"
    pts   = corners_pts.astype(int)

    # Draw filled polygon (semi-transparent)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Draw border — thicker if locked
    border = 3 if locked else 2
    cv2.polylines(frame, [pts], True, color, border)

    # Corner dots
    for pt in pts:
        cv2.circle(frame, tuple(pt), 4, color, -1)

    # ── Label ABOVE the topmost corner ──────────────────────────
    # Find the topmost y coordinate of the marker corners
    top_y  = int(pts[:, 1].min())   # smallest y = highest on screen
    top_x  = int(pts[:, 0].mean())  # horizontally centered on marker

    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.55, 1)
    label_x = top_x - tw // 2
    label_y = top_y - 10           # 10px gap above the marker top edge

    # Keep label inside frame
    label_x = max(2, min(frame.shape[1] - tw - 2, label_x))
    label_y = max(th + 4, label_y)

    # Background box
    cv2.rectangle(frame,
                  (label_x - 4, label_y - th - 4),
                  (label_x + tw + 4, label_y + 4),
                  (0, 0, 0), -1)
    # Lock indicator on background
    bg_color = (0, 120, 0) if locked else (0, 0, 0)
    cv2.rectangle(frame,
                  (label_x - 4, label_y - th - 4),
                  (label_x + tw + 4, label_y + 4),
                  bg_color, -1)
    cv2.putText(frame, label, (label_x, label_y), FONT_BOLD, 0.55, color, 1, cv2.LINE_AA)
    if locked:
        cv2.putText(frame, "✓", (label_x + tw + 6, label_y), FONT_BOLD, 0.45, (0, 255, 100), 1)

def draw_board_outline(frame, marker_centers):
    """Draw lines connecting the 4 markers to show the board boundary."""
    order = [10, 11, 12, 13, 10]  # TL→TR→BR→BL→TL
    for i in range(len(order) - 1):
        if order[i] in marker_centers and order[i+1] in marker_centers:
            p1 = tuple(marker_centers[order[i]].astype(int))
            p2 = tuple(marker_centers[order[i+1]].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

def build_status_panel(detected_ids, locked_ids, seen_counts, frame_count, fps):
    """Build the right-side status panel."""
    panel = np.zeros((DISPLAY_HEIGHT, STATUS_W, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)

    y = 20
    cv2.putText(panel, "CHESS ROBOT", (10, y), FONT_BOLD, 0.7, (200, 200, 255), 1)
    y += 22
    cv2.putText(panel, "CALIBRATION", (10, y), FONT_BOLD, 0.7, (200, 200, 255), 1)
    y += 30
    cv2.line(panel, (10, y), (STATUS_W - 10, y), (80, 80, 120), 1)
    y += 18

    cv2.putText(panel, "ARUCO MARKERS", (10, y), FONT, 0.45, (150, 150, 180), 1)
    y += 20

    for mid, cfg in MARKER_CONFIG.items():
        color   = cfg["color_bgr"]
        locked  = mid in locked_ids
        seen    = mid in detected_ids
        count   = seen_counts.get(mid, 0)

        # Status text + dot colour
        if locked:
            status    = f"LOCKED  ({count}x)"
            dot_color = color
            txt_color = (80, 230, 80)
        elif seen:
            status    = f"seen {count}/{MIN_SEEN_TO_LOCK}"
            dot_color = tuple(int(c * 0.6) for c in color)
            txt_color = (200, 180, 50)
        else:
            status    = "SEARCHING..."
            dot_color = (60, 60, 60)
            txt_color = (160, 80, 80)

        cv2.circle(panel, (22, y - 4), 7, dot_color, -1)
        if locked:
            cv2.circle(panel, (22, y - 4), 7, (255, 255, 255), 1)

        lbl = f"ID {mid:2d} ({cfg['label']})"
        cv2.putText(panel, lbl,    (38, y),      FONT, 0.5,  (220, 220, 220), 1)
        cv2.putText(panel, status, (38, y + 16), FONT, 0.38, txt_color, 1)
        y += 44

    y += 4
    cv2.line(panel, (10, y), (STATUS_W - 10, y), (80, 80, 120), 1)
    y += 18

    # Overall progress
    n_locked = len(locked_ids)
    prog_col = (80, 230, 80) if n_locked == 4 else (200, 140, 0) if n_locked >= 2 else (180, 80, 80)
    cv2.putText(panel, f"Locked: {n_locked}/4", (10, y), FONT_BOLD, 0.65, prog_col, 1)
    y += 26

    # Progress bar for locked markers
    bar_w = STATUS_W - 30
    cv2.rectangle(panel, (10, y), (10 + bar_w, y + 14), (40, 40, 60), -1)
    filled = int(bar_w * n_locked / 4)
    cv2.rectangle(panel, (10, y), (10 + filled, y + 14), prog_col, -1)
    y += 24

    if n_locked < 4:
        cv2.putText(panel, "Each marker locks", (10, y), FONT, 0.38, (150, 150, 180), 1)
        y += 15
        cv2.putText(panel, "independently!", (10, y), FONT, 0.38, (150, 150, 180), 1)
        y += 22
    else:
        cv2.putText(panel, "ALL LOCKED — saving...", (10, y), FONT, 0.42, (80, 230, 80), 1)
        y += 22

    cv2.line(panel, (10, y), (STATUS_W - 10, y), (80, 80, 120), 1)
    y += 15

    cv2.putText(panel, f"Frame: {frame_count}", (10, y), FONT, 0.42, (130, 130, 160), 1)
    y += 16
    cv2.putText(panel, f"FPS:   {fps:.1f}",    (10, y), FONT, 0.42, (130, 130, 160), 1)
    y += 26

    cv2.line(panel, (10, y), (STATUS_W - 10, y), (80, 80, 120), 1)
    y += 14
    cv2.putText(panel, "TIPS", (10, y), FONT, 0.42, (150, 150, 180), 1)
    y += 16
    for tip in ["Keep markers flat", "Even diffuse lighting",
                "Avoid glare on markers", "Camera ~50-80cm above",
                "Show 1 marker at a time OK!"]:
        cv2.putText(panel, f"• {tip}", (10, y), FONT, 0.35, (120, 130, 150), 1)
        y += 15

    y = DISPLAY_HEIGHT - 55
    cv2.line(panel, (10, y), (STATUS_W - 10, y), (80, 80, 120), 1)
    y += 15
    cv2.putText(panel, "Q quit | R reset | S save now", (10, y), FONT, 0.38, (100, 100, 130), 1)

    return panel

def build_display(frame, detected_centers, corners_raw, locked_centers,
                  locked_ids, seen_counts, frame_count, fps, warp_preview=None):
    """Composite the annotated frame + warp preview + status panel."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw board outline using locked centers (stable) or detected (live)
    display_centers = {**detected_centers, **locked_centers}
    if len(display_centers) >= 2:
        draw_board_outline(annotated, display_centers)

    # Draw locked markers (from accumulated cache) — dimmed outline
    for mid, center in locked_centers.items():
        if mid in corners_raw:
            draw_marker_on_frame(annotated, mid, corners_raw[mid], center, locked=True)
        elif mid not in detected_centers:
            # Draw just a dot at locked position if not currently visible
            cfg = MARKER_CONFIG[mid]
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(annotated, (cx, cy), 10, cfg["color_bgr"], 2)
            cv2.putText(annotated, f"ID{mid} ✓", (cx - 20, cy - 14),
                        FONT_BOLD, 0.45, cfg["color_bgr"], 1)

    # Draw currently detected markers (live this frame)
    for mid, center in detected_centers.items():
        if mid in corners_raw:
            locked = mid in locked_ids
            draw_marker_on_frame(annotated, mid, corners_raw[mid], center, locked=locked)

    # Header bar
    n_locked = len(locked_ids)
    if n_locked == 4:
        hdr_color = (0, 180, 60)
        hdr_txt   = "ArUco Calibration | 4/4 LOCKED — saving..."
    else:
        hdr_color = (0, 180, 220)
        hdr_txt   = f"ArUco Calibration | {n_locked}/4 locked | {len(detected_centers)}/4 visible this frame"
    cv2.rectangle(annotated, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(annotated, hdr_txt, (8, 22), FONT, 0.58, hdr_color, 1, cv2.LINE_AA)

    # White-side label
    cv2.rectangle(annotated, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(annotated, "WHITE SIDE  ▼", (w//2 - 55, h - 6), FONT, 0.5, (200, 200, 200), 1)

    scale       = DISPLAY_HEIGHT / h
    new_w       = int(w * scale)
    annotated_r = cv2.resize(annotated, (new_w, DISPLAY_HEIGHT))

    panel = build_status_panel(set(detected_centers.keys()), locked_ids, seen_counts, frame_count, fps)

    # Warp preview
    if warp_preview is not None:
        pw       = STATUS_W - 20
        py_start = DISPLAY_HEIGHT - pw - 70
        if py_start > 0:
            warp_small = cv2.resize(warp_preview, (pw, pw))
            panel[py_start:py_start + pw, 10:10 + pw] = warp_small
            cv2.rectangle(panel, (10, py_start), (10 + pw, py_start + pw), (0, 255, 255), 1)
            cv2.putText(panel, "Warp Preview", (10, py_start - 6), FONT, 0.38, (0, 220, 220), 1)

    return np.hstack([annotated_r, panel])

# ── Cache I/O ─────────────────────────────────────────────────────────────────
def save_cache(marker_centers, warp_matrix):
    """Save detected board corners and warp matrix to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "marker_centers": {
            str(mid): [float(c[0]), float(c[1])]
            for mid, c in marker_centers.items()
        },
        "warp_matrix": warp_matrix.tolist(),
        "warp_size": WARP_SIZE,
        "notes": "Generated by aruco_calibration.py — delete to re-calibrate",
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  ✅  CACHE SAVED → {os.path.abspath(CACHE_FILE)}")
    print(f"{'='*60}")

def load_cache():
    """Load existing cache if available."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None

# ── Redis ─────────────────────────────────────────────────────────────────────
def connect_redis():
    r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r_bin.ping()
    return r_bin, r_str

def read_frame_from_redis(r_bin, last_frame_id, r_str):
    """Returns (frame, frame_id) or (None, last_frame_id) if no new frame."""
    try:
        new_id = r_str.get("latest_frame_id")
        if new_id is None or new_id == last_frame_id:
            return None, last_frame_id
        raw = r_bin.get("latest_frame")
        if raw is None:
            return None, last_frame_id
        img_np = np.frombuffer(raw, dtype=np.uint8)
        frame  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return frame, new_id
    except Exception as e:
        print(f"⚠️  Redis read error: {e}")
        return None, last_frame_id

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ♟  Chess Robot — ArUco Board Calibration")
    print("=" * 60)

    # ── Redis connect ──────────────────────────────────────────
    print("\n🔌 Connecting to Redis...")
    try:
        r_bin, r_str = connect_redis()
        print("✅ Redis OK\n")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Make sure: redis-server is running")
        print("   And: camera_capture_android.py is running in Terminal 1")
        sys.exit(1)

    # ── Check if cache already exists ─────────────────────────
    existing = load_cache()
    if existing:
        print(f"📂 Existing cache found: {CACHE_FILE}")
        print(f"   Saved at: {existing.get('timestamp', 'unknown')}")
        ans = input("   Use existing cache and skip calibration? [Y/n]: ").strip().lower()
        if ans != "n":
            print("\n✅ Using cached board corners.")
            print_next_step()
            return

    # ── Wait for frames ────────────────────────────────────────
    print("⏳ Waiting for frames from camera_capture_android.py...")
    print("   (Make sure Terminal 1 is running that script)\n")

    timeout_start = time.time()
    while True:
        frame_id = r_str.get("latest_frame_id")
        if frame_id is not None:
            print(f"✅ Frames arriving (frame_id = {frame_id})\n")
            break
        if time.time() - timeout_start > 30:
            print("❌ Timed out waiting for frames. Is camera_capture_android.py running?")
            sys.exit(1)
        time.sleep(0.5)

    # ── Calibration loop ───────────────────────────────────────
    print("📐 Starting calibration — markers lock independently across frames")
    print("   No need to show all 4 at once! Each locks when seen enough times.")
    print("   Press Q to quit | R to reset | S to force-save all locked\n")

    # ── Accumulation state ─────────────────────────────────────
    # Each marker locks individually — no need for all 4 in same frame
    seen_counts     = {}   # mid → how many times detected so far
    sum_centers     = {}   # mid → sum of center positions (for averaging)
    sum_corners     = {}   # mid → latest corner points
    locked_ids      = set()   # markers that have reached MIN_SEEN_TO_LOCK
    locked_centers  = {}   # mid → final averaged center (locked in)
    locked_corners  = {}   # mid → corners at lock time

    frame_count    = 0
    last_frame_id  = None
    fps            = 0.0
    t_fps          = time.time()
    fps_frames     = 0
    last_warp_preview = None
    calibration_done  = False

    cv2.namedWindow("Chess Robot — Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Chess Robot — Calibration", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while not calibration_done:
        frame, last_frame_id = read_frame_from_redis(r_bin, last_frame_id, r_str)

        if frame is None:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit by user.")
                break
            continue

        frame_count += 1
        fps_frames  += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps        = fps_frames / (now - t_fps)
            fps_frames = 0
            t_fps      = now

        # ── Detect markers in this frame ───────────────────────
        detected_centers, corners_raw = detect_aruco_markers(frame)

        # ── Accumulate each detected marker ────────────────────
        for mid, center in detected_centers.items():
            if mid in locked_ids:
                continue   # already locked — don't update

            seen_counts[mid] = seen_counts.get(mid, 0) + 1

            # Running sum for averaging
            if mid not in sum_centers:
                sum_centers[mid] = center.copy()
            else:
                sum_centers[mid] = sum_centers[mid] + center

            # Keep latest corners for drawing
            if mid in corners_raw:
                sum_corners[mid] = corners_raw[mid]

            # Lock when seen enough times
            if seen_counts[mid] >= MIN_SEEN_TO_LOCK and mid not in locked_ids:
                locked_ids.add(mid)
                locked_centers[mid] = sum_centers[mid] / seen_counts[mid]
                locked_corners[mid] = sum_corners[mid]
                cfg = MARKER_CONFIG[mid]
                print(f"  🔒 ID{mid} ({cfg['label']}) LOCKED after {seen_counts[mid]} detections "
                      f"— center: ({locked_centers[mid][0]:.1f}, {locked_centers[mid][1]:.1f})")

        # ── Warp preview using locked centers (stable) ─────────
        warp_preview = last_warp_preview
        all_centers_for_warp = {**locked_centers}
        # Fill in live detected for any not yet locked (preview only)
        for mid, c in detected_centers.items():
            if mid not in all_centers_for_warp:
                all_centers_for_warp[mid] = c

        if len(all_centers_for_warp) == 4:
            try:
                M, _, _ = compute_warp_matrix(all_centers_for_warp)
                warp_preview      = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))
                last_warp_preview = warp_preview
            except Exception:
                pass

        # ── Build & show display ───────────────────────────────
        display = build_display(
            frame, detected_centers, corners_raw,
            locked_centers, locked_ids, seen_counts,
            frame_count, fps, warp_preview
        )
        cv2.imshow("Chess Robot — Calibration", display)

        # ── Key handling ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 Quit by user.")
            break
        elif key == ord('r'):
            seen_counts    = {}
            sum_centers    = {}
            sum_corners    = {}
            locked_ids     = set()
            locked_centers = {}
            locked_corners = {}
            last_warp_preview = None
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            print("🔄 Reset — all markers unlocked, starting fresh")
        elif key == ord('s') and len(locked_ids) == 4:
            M, _, _ = compute_warp_matrix(locked_centers)
            save_cache(locked_centers, M)
            calibration_done = True
            break

        # ── Auto-save when all 4 locked ────────────────────────
        if len(locked_ids) == 4:
            print("\n✅ All 4 markers locked! Saving cache...")

            # Flash the screen
            flash = display.copy()
            h_f, w_f = flash.shape[:2]
            cv2.rectangle(flash,
                          (w_f//2 - 220, h_f//2 - 40),
                          (w_f//2 + 220, h_f//2 + 40),
                          (0, 180, 60), -1)
            cv2.putText(flash, "ALL 4 LOCKED — SAVED!",
                        (w_f//2 - 200, h_f//2 + 14),
                        FONT_BOLD, 0.85, (255, 255, 255), 2)
            cv2.imshow("Chess Robot — Calibration", flash)
            cv2.waitKey(1500)

            M, _, _ = compute_warp_matrix(locked_centers)
            save_cache(locked_centers, M)
            calibration_done = True
            break

    cv2.destroyAllWindows()

    if calibration_done:
        print_next_step()
    else:
        print("\n⚠️  Calibration not completed. Run again when markers are visible.")

def print_next_step():
    print()
    print("╔" + "═"*58 + "╗")
    print("║  🎉  CALIBRATION COMPLETE — Board corners cached!      ║")
    print("╠" + "═"*58 + "╣")
    print("║                                                          ║")
    print("║  Open a NEW terminal and run:                            ║")
    print("║                                                          ║")
    print("║      python main.py                                      ║")
    print("║                                                          ║")
    print("║  main.py will automatically load board_cache.json        ║")
    print("║  and skip ArUco detection during the game.               ║")
    print("║                                                          ║")
    print("╚" + "═"*58 + "╝")
    print()

if __name__ == "__main__":
    main()