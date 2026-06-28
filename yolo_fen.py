# """
# yolo_fen.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chess Robot — Vision Engine (Corrected Warp + Step-by-Step Visualisation)

# HOW IT WORKS:
#   1. Load 4 marker positions (IDs 12, 13, 11, 10) from board_cache.json.
#   2. Scale markers to match the current frame resolution.
#   3. Build warp matrix M from these scaled markers.
#   4. Per frame:
#        a. Run YOLO on the ORIGINAL image (best detections).
#        b. Transform each piece center -> board space via M.
#        c. Map board-space pixels to 8x8 squares.
#        d. Build FEN string.
#        e. Show step-by-step visual feedback.
# """

# import cv2
# import numpy as np
# import json
# import os
# import logging
# import time

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ── CONFIG ────────────────────────────────────────────────────────────────────
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_finetuned.pt")
# CACHE_FILE = "board_cache.json"
# BOARD_SIZE = 800          # Size of the warped board image (pixels)

# # MARGIN: The markers - Adjust this until the 8x8 grid perfectly fills the 800x800 warped image.
# # 0.05 (5%) is a good balanced margin for this board setup.
# BOARD_MARGIN = 0.05

# # Use floating square size for perfect grid alignment (prevents integer rounding drift)
# SQ_SIZE = BOARD_SIZE / 8.0

# FEN_MAP = {
#     "white_pawn":   "P", "black_pawn":   "p",
#     "white_rook":   "R", "black_rook":   "r",
#     "white_knight": "N", "black_knight": "n",
#     "white_bishop": "B", "black_bishop": "b",
#     "white_queen":  "Q", "black_queen": "q",
#     "white_king":   "K", "black_king":   "k",
# }

# # ══════════════════════════════════════════════════════════════════════════════
# # LOAD RESOURCES
# # ══════════════════════════════════════════════════════════════════════════════

# def _load_yolo_model():
#     from ultralytics import YOLO
#     if not os.path.exists(MODEL_PATH):
#         logger.error(f"Model file not found at {MODEL_PATH}")
#         return None
#     m = YOLO(MODEL_PATH)
#     logger.info(f" YOLO Model loaded: {MODEL_PATH}")
#     return m

# def _load_calibration_data():
#     if not os.path.exists(CACHE_FILE):
#         return None
#     with open(CACHE_FILE) as f:
#         return json.load(f)

# # Global model and cache (lazy load)
# _MODEL = None
# _CACHE_DATA = None

# # ══════════════════════════════════════════════════════════════════════════════
# # PROCESSING PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════

# def process_frame(frame_input, frame_id=0, visualise: bool = True) -> dict:
#     global _MODEL, _CACHE_DATA
    
#     if _MODEL is None:
#         _MODEL = _load_yolo_model()
#     if _CACHE_DATA is None:
#         _CACHE_DATA = _load_calibration_data()
        
#     if _CACHE_DATA is None:
#         return _fail("board_cache.json not found. Run aruco_calibration.py first.")

#     # 1. Decode Frame
#     if isinstance(frame_input, bytes):
#         arr = np.frombuffer(frame_input, np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     elif isinstance(frame_input, str):
#         img = cv2.imread(frame_input)
#     else:
#         img = frame_input

#     if img is None:
#         return _fail("Could not read/decode frame.")

#     h, w = img.shape[:2]
    
#     # 2. Extract and Scale Markers
#     # Correct Mapping from aruco_calibration.py:
#     # 12: TL, 13: TR, 11: BR, 10: BL
#     try:
#         centers = _CACHE_DATA["marker_centers"]
#         tl = np.array(centers["12"])
#         tr = np.array(centers["13"])
#         br = np.array(centers["11"])
#         bl = np.array(centers["10"])
        
#         # Scale if calibration resolution differs from current frame
#         if "source_size" in _CACHE_DATA and _CACHE_DATA["source_size"] is not None:
#             calib_w, calib_h = _CACHE_DATA["source_size"]
#             scale_x = w / calib_w
#             scale_y = h / calib_h
#             tl = tl * [scale_x, scale_y]
#             tr = tr * [scale_x, scale_y]
#             br = br * [scale_x, scale_y]
#             bl = bl * [scale_x, scale_y]
            
#         src_pts = np.float32([tl, tr, br, bl])
        
#         # PERFECT 64 SQUARES: Zoom in so the grid fills the 800x800 view
#         # We map markers to a larger virtual area so the inner 8x8 grid 
#         # fits exactly into the 800x800 output.
#         total_v_size = BOARD_SIZE / (1 - 2 * BOARD_MARGIN)
#         offset = (total_v_size - BOARD_SIZE) / 2
        
#         dst_pts = np.float32([
#             [-offset, -offset], 
#             [BOARD_SIZE + offset, -offset], 
#             [BOARD_SIZE + offset, BOARD_SIZE + offset], 
#             [-offset, BOARD_SIZE + offset]
#         ])
        
#         M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     except KeyError as e:
#         return _fail(f"Missing marker ID in cache: {e}. Re-calibrate.")

#     # 3. Detect Pieces on ORIGINAL image
#     results = _MODEL.predict(img, conf=0.15, iou=0.45, imgsz=1024, verbose=False)
#     boxes = results[0].boxes
    
#     square_detections = {}
    
#     for box in boxes:
#         cls = int(box.cls[0])
#         name = _MODEL.names[cls]
#         conf = float(box.conf[0])
        
#         if name not in FEN_MAP:
#             continue
            
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
        
#         # FIXED: Center of the piece base (slightly up from the absolute bottom)
#         h_px = y2 - y1
#         cx = (x1 + x2) / 2
#         # FIXED: Center of the piece base (15% up to avoid the very front edge)
#         cy = y2 - (h_px * 0.15)
        
#         # INWARD SHIFT: Pull the point slightly towards the center of the board.
#         # This compensates for the camera's perspective lean.
#         img_h, img_w = img.shape[:2]
#         cx = cx + (img_w/2 - cx) * 0.05  # Pull 5% towards center X
#         cy = cy + (img_h/2 - cy) * 0.05  # Pull 5% towards center Y
        
#         # Transform to board space
#         pt = np.array([[[cx, cy]]], dtype=np.float32)
#         pt_board = cv2.perspectiveTransform(pt, M)[0][0]
#         bx, by = pt_board[0], pt_board[1]
        
#         # 3. Calculate mapping (Warped image is now exactly 8x8)
#         # Check if inside board boundaries [0, BOARD_SIZE]
#         if -20 <= bx < BOARD_SIZE + 20 and -20 <= by < BOARD_SIZE + 20:
            
#             col = int(np.clip(bx // SQ_SIZE, 0, 7))
#             row = int(np.clip(by // SQ_SIZE, 0, 7))
            
#             # Highest confidence wins the square
#             if (row, col) not in square_detections or conf > square_detections[(row, col)]["conf"]:
#                 square_detections[(row, col)] = {
#                     "char": FEN_MAP[name],
#                     "name": name,
#                     "conf": conf,
#                     "box": (x1, y1, x2, y2),
#                     "board_pt": (int(bx), int(by))
#                 }
#         else:
#             logger.debug(f"Piece {name} rejected at board coords ({bx:.1f}, {by:.1f})")

#     # 4. Generate FEN
#     fen_rows = []
#     for r in range(8):
#         row_str = ""
#         empty = 0
#         for c in range(8):
#             if (r, c) in square_detections:
#                 if empty > 0:
#                     row_str += str(empty)
#                     empty = 0
#                 row_str += square_detections[(r, c)]["char"]
#             else:
#                 empty += 1
#         if empty > 0:
#             row_str += str(empty)
#         fen_rows.append(row_str)
    
#     fen = "/".join(fen_rows) + " w - - 0 1"
    
#     # 5. Result Summary
#     logger.info(f"Frame {frame_id} | FEN: {fen} | Pieces: {len(square_detections)}")
    
#     # 6. Step-by-Step Visualization
#     if visualise:
#         _visualise_results(img, M, square_detections, fen, frame_id)

#     return {
#         "success": True,
#         "fen": fen,
#         "pieces_count": len(square_detections),
#         "frame_id": frame_id
#     }
# def _visualise_results(img, M, detections, fen, frame_id):
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches
    
#     # Generate Warped Board
#     warped = cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))
    
#     fig, axes = plt.subplots(1, 3, figsize=(20, 7))
#     fig.canvas.manager.set_window_title(f"FEN Generator - Frame {frame_id}")
    
#     # Panel 1: Original + YOLO
#     axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     axes[0].set_title(f"Step 1: YOLO Detection (Frame {frame_id})")
#     for (r, c), d in detections.items():
#         x1, y1, x2, y2 = d["box"]
#         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
#         axes[0].add_patch(rect)
#         axes[0].text(x1, y1-5, f"{d['char']} ({d['conf']:.2f})", color='yellow', fontsize=10, weight='bold', bbox=dict(facecolor='black', alpha=0.5))
#     axes[0].axis('off')

#     # Panel 2: Warped Top-Down
#     axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
#     axes[1].set_title("Step 2: Correct Perspective Warp")

#     # Draw perfect 64 squares (filling the entire zoomed view) using precise float positions
#     grid_positions = list(np.linspace(0.0, float(BOARD_SIZE), 9))
#     for pos in grid_positions:
#         # Horizontal lines
#         axes[1].plot([0.0, BOARD_SIZE], [pos, pos], color='lime', alpha=0.8, linewidth=1)
#         # Vertical lines
#         axes[1].plot([pos, pos], [0.0, BOARD_SIZE], color='lime', alpha=0.8, linewidth=1)

#     # Draw transformed piece centers (use the exact board-space point we computed earlier)
#     for (r, c), d in detections.items():
#         bx, by = d.get("board_pt", (None, None))
#         if bx is None:
#             # fallback to cell center if board_pt missing
#             bx = (c + 0.5) * SQ_SIZE
#             by = (r + 0.5) * SQ_SIZE

#         # Clip to visible warped area for plotting
#         bx = float(np.clip(bx, 0.0, BOARD_SIZE))
#         by = float(np.clip(by, 0.0, BOARD_SIZE))

#         axes[1].plot(bx, by, 'yo', markersize=8, markeredgecolor='black')
#         # Draw the detected piece label near its transformed center; choose contrasting color
#         label_color = 'black' if d['char'].isupper() else 'white'
#         axes[1].text(bx + 4, by - 6, d["char"], color=label_color, weight='bold', fontsize=10,
#                      bbox=dict(facecolor='black' if label_color=='white' else 'white', alpha=0.6))
#     axes[1].axis('off')

#     # Panel 3: FEN Generation Result (FIXED FLIP)
#     axes[2].set_title(f"Step 3: Final FEN Output\n{fen}", fontsize=10)
#     axes[2].set_xlim(0, 8)
#     axes[2].set_ylim(0, 8)
    
#     for r in range(8):
#         for c in range(8):
#             # Standard chess pattern logic adjusted so the bottom-left (a1) is the light square
#             color = '#f0d9b5' if ((7 - r) + c) % 2 == 0 else '#b58863'

#             # FIXED: 7-r maps image row 0 (top) to plot row 7 (top)
#             rect = patches.Rectangle((c, 7 - r), 1, 1, facecolor=color, edgecolor='black', alpha=0.8)
#             axes[2].add_patch(rect)
            
#             if (r, c) in detections:
#                 char = detections[(r, c)]["char"]
#                 # White pieces are uppercase, black are lowercase
#                 txt_color = 'black' if char.isupper() else 'white'
#                 axes[2].text(c + 0.5, 7 - r + 0.5, char, ha='center', va='center', fontsize=20, weight='bold', color=txt_color)
    
#     # Set coordinates matching a standard board layout where rank 1 is at the bottom
#     axes[2].set_xticks(np.arange(0.5, 8.5, 1))
#     axes[2].set_xticklabels(['a','b','c','d','e','f','g','h'])
#     axes[2].set_yticks(np.arange(0.5, 8.5, 1))
#     axes[2].set_yticklabels(['1','2','3','4','5','6','7','8']) # 1 is bottom, 8 is top
    
#     plt.tight_layout()
#     plt.show()

# def _fail(msg):
#     logger.error(f"❌ {msg}")
#     return {"success": False, "error": msg}

# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN TEST BLOCK
# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     import sys
    
#     # Example usage: python yolo_fen.py path/to/image.jpg
#     if len(sys.argv) > 1:
#         path = sys.argv[1]
#         process_frame(path, frame_id=1, visualise=True)
#     else:
#         # Check for images folder
#         if os.path.exists("images"):
#             files = sorted([os.path.join("images", f) for f in os.listdir("images") if f.endswith(('.jpg', '.png'))])
#             if files:
#                 print(f"Processing first image found: {files[0]}")
#                 process_frame(files[0], frame_id=1, visualise=True)
#             else:
#                 print("No images found in 'images' folder.")
#         else:
#             print("Usage: python yolo_fen.py <image_path>")

"""
yolo_fen.py  -- Chess Robot Vision Engine  v4 (Perspective Centroid Fix)
========================================================================
ROOT CAUSE OF PREVIOUS WARP BUG FIXED:
  For side-angle cameras, tracking a single point at the bottom of a 3D 
  piece introduces perspective drift toward the near edge of its cell.
  
THE FIX:
  We extract all 4 corners of the piece's bounding box, project them all into
  the 2D top-down board space using the homography matrix M, and calculate
  their centroid. This accurately centers the piece over its designated 
  square regardless of camera skew or piece height.
"""

import cv2
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PATHS ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_finetuned.pt")
CACHE_FILE = "board_cache.json"

# ── WARP -- must match aruco_calibration.py ───────────────────────────────────
DEFAULT_WARP_SIZE = 640   # aruco_calibration.py default

# Corner order for getPerspectiveTransform -- MATCHES aruco_calibration.py:
#   compute_warp_matrix uses:  12->TL  13->TR  11->BR  10->BL
ARUCO_CORNER_ORDER = [12, 13, 11, 10]

# Reuse live-detected H for this many frames when markers are hidden
HOMOGRAPHY_MAX_AGE = 30

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_CONF  = 0.15
YOLO_IOU   = 0.45
YOLO_IMGSZ = 1024

# ── FEN MAP ───────────────────────────────────────────────────────────────────
FEN_MAP = {
    "white_pawn":   "P", "black_pawn":   "p",
    "white_rook":   "R", "black_rook":   "r",
    "white_knight": "N", "black_knight": "n",
    "white_bishop": "B", "black_bishop": "b",
    "white_queen":  "Q", "black_queen":  "q",
    "white_king":   "K", "black_king":   "k",
}

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
_MODEL      = None
_CACHE_DATA = None
_H_CACHE    = {"M": None, "age": 0, "wsize": DEFAULT_WARP_SIZE}


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        _MODEL = YOLO(MODEL_PATH)
        logger.info(f"YOLO loaded: {MODEL_PATH}")
    return _MODEL


def _cache():
    global _CACHE_DATA
    if _CACHE_DATA is None:
        if not os.path.exists(CACHE_FILE):
            return None
        with open(CACHE_FILE) as f:
            _CACHE_DATA = json.load(f)
        logger.info(f"Cache loaded: {CACHE_FILE}")
    return _CACHE_DATA


# ══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════

def _warp_size():
    c = _cache()
    return int(c["warp_size"]) if c and "warp_size" in c else DEFAULT_WARP_SIZE


def _dst_pts(wsize):
    """Destination corners matching aruco_calibration.py: TL TR BR BL."""
    W = wsize
    return np.float32([[0, 0], [W-1, 0], [W-1, W-1], [0, W-1]])


def _try_live_aruco(gray, wsize):
    """
    Try same ArUco dictionaries as aruco_calibration.py.
    Returns homography M or None.
    """
    dicts = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_6X6_250,
    ]
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    images  = [gray, clahe.apply(gray)]
    targets = set(ARUCO_CORNER_ORDER)
    best    = {}

    for img_c in images:
        for dict_id in dicts:
            d  = cv2.aruco.getPredefinedDictionary(dict_id)
            p  = cv2.aruco.DetectorParameters()
            det = cv2.aruco.ArucoDetector(d, p)
            corners, ids, _ = det.detectMarkers(img_c)
            if ids is None:
                continue
            found = {}
            for i, mid in enumerate(ids.flatten()):
                if mid in targets:
                    found[int(mid)] = corners[i][0].mean(axis=0)
            if len(found) > len(best):
                best = found
            if len(best) == 4:
                break
        if len(best) == 4:
            break

    if len(best) < 4:
        return None

    src = np.float32([best[mid] for mid in ARUCO_CORNER_ORDER])
    return cv2.getPerspectiveTransform(src, _dst_pts(wsize))


def _get_homography(img):
    """
    Returns (M, warp_size, source_label).
    Priority:
      1. Live ArUco detection every frame
      2. Recently cached live H
      3. warp_matrix directly from board_cache.json
    """
    wsize = _warp_size()
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Live ArUco
    M_live = _try_live_aruco(gray, wsize)
    if M_live is not None:
        _H_CACHE.update({"M": M_live, "age": 0, "wsize": wsize})
        return M_live, wsize, "live-aruco"

    # 2. Cached live H
    if _H_CACHE["M"] is not None and _H_CACHE["age"] < HOMOGRAPHY_MAX_AGE:
        _H_CACHE["age"] += 1
        return _H_CACHE["M"], _H_CACHE["wsize"], f"cached(age={_H_CACHE['age']})"

    # 3. Saved matrix from aruco_calibration.py
    c = _cache()
    if c and "warp_matrix" in c:
        M_file  = np.array(c["warp_matrix"], dtype=np.float64)
        wsize_f = int(c.get("warp_size", DEFAULT_WARP_SIZE))
        return M_file, wsize_f, "cache-file"

    return None, wsize, "none"


# ══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE CENTROID TRANSFORM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_to_board_centroid(x1, y1, x2, y2, M):
    """
    Transforms all 4 corners of a bounding box into top-down board coordinates,
    and returns the centroid of that projected space.
    """
    # 1. Define the 4 corners of the bbox in image space
    corners = np.float32([
        [x1, y1],  # Top-Left
        [x2, y1],  # Top-Right
        [x2, y2],  # Bottom-Right
        [x1, y2]   # Bottom-Left
    ]).reshape(-1, 1, 2)
    
    # 2. Transform all corners simultaneously into board space
    projected_corners = cv2.perspectiveTransform(corners, M)
    projected_corners = projected_corners.reshape(-1, 2)
    
    # 3. Find the arithmetic mean (centroid) of the projected bounding region
    bx, by = np.mean(projected_corners, axis=0)
    return float(bx), float(by)


def _to_square(bx, by, wsize, tol=20):
    sq = wsize / 8.0
    if not (-tol <= bx < wsize + tol and -tol <= by < wsize + tol):
        return None
    return int(np.clip(by // sq, 0, 7)), int(np.clip(bx // sq, 0, 7))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame_input, frame_id=0, visualise=True):
    # Decode frame
    if isinstance(frame_input, bytes):
        img = cv2.imdecode(np.frombuffer(frame_input, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(frame_input, str):
        img = cv2.imread(frame_input)
    else:
        img = frame_input.copy()

    if img is None:
        return _fail("Cannot read frame.")

    # Get homography
    M, wsize, h_src = _get_homography(img)
    if M is None:
        return _fail("No homography. Run aruco_calibration.py first.")

    # YOLO detect
    results = _model().predict(img, conf=YOLO_CONF, iou=YOLO_IOU,
                               imgsz=YOLO_IMGSZ, verbose=False)

    square_detections = {}
    for box in results[0].boxes:
        cls  = int(box.cls[0])
        name = _model().names[cls]
        conf = float(box.conf[0])
        if name not in FEN_MAP:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # FIXED: Project all corners & extract perspective-correct center
        bx, by = _bbox_to_board_centroid(x1, y1, x2, y2, M)
        
        sq = _to_square(bx, by, wsize)
        if sq is None:
            continue
        row, col = sq
        if (row, col) not in square_detections or conf > square_detections[(row, col)]["conf"]:
            square_detections[(row, col)] = {
                "char":     FEN_MAP[name],
                "name":     name,
                "conf":     conf,
                "box":      (x1, y1, x2, y2),
                "board_pt": (int(bx), int(by)),
            }

    # Build FEN
    fen_rows = []
    for r in range(8):
        s, empty = "", 0
        for c in range(8):
            if (r, c) in square_detections:
                if empty:
                    s += str(empty)
                    empty = 0
                s += square_detections[(r, c)]["char"]
            else:
                empty += 1
        if empty:
            s += str(empty)
        fen_rows.append(s)

    fen = "/".join(fen_rows) + " w - - 0 1"
    logger.info(f"Frame {frame_id} | {h_src} | pieces={len(square_detections)} | {fen}")

    if visualise:
        _visualise(img, M, wsize, square_detections, fen, frame_id, h_src)

    return {"success": True, "fen": fen,
            "pieces_count": len(square_detections),
            "frame_id": frame_id, "h_source": h_src}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _visualise(img, M, wsize, detections, fen, frame_id, h_src):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    warped = cv2.warpPerspective(img, M, (wsize, wsize))
    sq     = wsize / 8.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"Frame {frame_id}  |  H={h_src}  |  Method: Projected Centroid  |  warp={wsize}px",
        fontsize=10)
    fig.canvas.manager.set_window_title(f"FEN -- Frame {frame_id}")

    # Panel 1: original + boxes
    ax = axes[0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Step 1: YOLO detections")
    for (r, c), d in detections.items():
        x1, y1, x2, y2 = d["box"]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, lw=1.5, edgecolor="red", facecolor="none"))
        ax.text(x1, y1-4, f"{d['char']}({d['conf']:.2f})",
                color="yellow", fontsize=7, weight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
    ax.axis("off")

    # Panel 2: warped board
    ax = axes[1]
    ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Step 2: Warped via {h_src}\nYellow dot = perspective-correct centroid")
    grid = np.linspace(0, wsize, 9)
    for p in grid:
        ax.plot([0, wsize], [p, p], color="lime", lw=0.8, alpha=0.7)
        ax.plot([p, p],     [0, wsize], color="lime", lw=0.8, alpha=0.7)
    for i in range(8):
        mid = (i + 0.5) * sq
        ax.text(mid, wsize-3, "abcdefgh"[i], color="cyan", ha="center", va="bottom", fontsize=6)
        ax.text(3,   mid,    str(8-i),      color="cyan", ha="left",   va="center", fontsize=6)
    for (r, c), d in detections.items():
        bx = float(np.clip(d["board_pt"][0], 0, wsize))
        by = float(np.clip(d["board_pt"][1], 0, wsize))
        ax.plot(bx, by, "yo", markersize=8, markeredgecolor='black')
        lc = "black" if d["char"].isupper() else "white"
        ax.text(bx+5, by-5, d["char"], color=lc, fontsize=9, weight="bold",
                bbox=dict(facecolor="white" if lc=="black" else "black", alpha=0.5))
    ax.axis("off")

    # Panel 3: FEN board
    ax = axes[2]
    ax.set_title(f"Step 3: FEN\n{fen}", fontsize=9)
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    for r in range(8):
        for c in range(8):
            col_sq = "#f0d9b5" if ((7-r)+c) % 2 == 0 else "#b58863"
            ax.add_patch(patches.Rectangle(
                (c, 7-r), 1, 1, facecolor=col_sq, edgecolor="#444", lw=0.4))
            if (r, c) in detections:
                ch = detections[(r, c)]["char"]
                tc = "black" if ch.isupper() else "white"
                ax.text(c+0.5, 7-r+0.5, ch, ha="center", va="center",
                        fontsize=18, weight="bold", color=tc)
    ax.set_xticks(np.arange(0.5, 8.5)); ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticks(np.arange(0.5, 8.5)); ax.set_yticklabels([str(i) for i in range(1, 9)])
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE TUNER -- REPLACED BY CENTROID TESTER
# ══════════════════════════════════════════════════════════════════════════════

def test_centroid_mapping(image_path):
    """
    Static OpenCV verification window. 
    Displays the output utilizing the brand-new bounding box centroid method.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}"); return

    M, wsize, src = _get_homography(img)
    if M is None:
        print("No homography -- run aruco_calibration.py first."); return

    mdl     = _model()
    results = mdl.predict(img, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False)
    
    vis = img.copy()
    warped = cv2.warpPerspective(img, M, (wsize, wsize))
    sq = wsize / 8.0
    
    # Draw Grids on top-down view
    for i in range(9):
        p = int(i * sq)
        cv2.line(warped, (p, 0), (p, wsize), (0, 255, 0), 1)
        cv2.line(warped, (0, p), (wsize, p), (0, 255, 0), 1)

    for box in results[0].boxes:
        cls  = int(box.cls[0])
        name = mdl.names[cls]
        if name not in FEN_MAP:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        bx, by = _bbox_to_board_centroid(x1, y1, x2, y2, M)
        sq_pos = _to_square(bx, by, wsize)
        
        color = (0, 255, 255) if sq_pos else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        wbx = int(np.clip(bx, 0, wsize-1))
        wby = int(np.clip(by, 0, wsize-1))
        cv2.circle(warped, (wbx, wby), 5, color, -1)

    cv2.imshow("Centroid Method Output Validation", np.hstack([
        cv2.resize(vis, (640, 480)),
        cv2.resize(warped, (480, 480))
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _fail(msg):
    logger.error(f"ERROR: {msg}")
    return {"success": False, "error": msg}


def print_cache_info():
    c = _cache()
    if c is None:
        print("board_cache.json not found."); return
    print(f"  timestamp  : {c.get('timestamp', 'n/a')}")
    print(f"  warp_size  : {c.get('warp_size', 'n/a')}")
    print(f"  source_size: {c.get('source_size', 'n/a')}")
    print(f"  warp_matrix: {'present' if 'warp_matrix' in c else 'MISSING -- re-run aruco_calibration.py'}")
    print(f"  markers    : {list(c.get('marker_centers', {}).keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "tune" and len(sys.argv) > 2:
            test_centroid_mapping(sys.argv[2])
        else:
            # fallback treat argument as direct image tracking path
            process_frame(cmd, frame_id=1, visualise=True)
    else:
        if os.path.exists("images"):
            files = sorted([os.path.join("images", f) for f in os.listdir("images") if f.endswith(('.jpg', '.png'))])
            if files:
                print(f"Processing first image found: {files[0]}")
                process_frame(files[0], frame_id=1, visualise=True)
            else:
                print("No images found in 'images' folder.")
        else:
            print("Usage: python yolo_fen.py <image_path>  OR  python yolo_fen.py tune <image_path>")