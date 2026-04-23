"""
yolo_fen.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess Robot — Vision Engine (Corrected Warp + Step-by-Step Visualisation)

HOW IT WORKS:
  1. Load 4 marker positions (IDs 12, 13, 11, 10) from board_cache.json.
  2. Scale markers to match the current frame resolution.
  3. Build warp matrix M from these scaled markers.
  4. Per frame:
       a. Run YOLO on the ORIGINAL image (best detections).
       b. Transform each piece center -> board space via M.
       c. Map board-space pixels to 8x8 squares.
       d. Build FEN string.
       e. Show step-by-step visual feedback.
"""

import cv2
import numpy as np
import json
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\shiza\OneDrive\Desktop\fyp_sw_test\best_finetuned.pt"
CACHE_FILE = "board_cache.json"
BOARD_SIZE = 800          # Size of the warped board image (pixels)

# MARGIN: The markers are on the board frame, not the exact corners of the a1-h8 grid.
# Increase this if pieces are shifted to the right/left. 0.05 = 5% margin.
BOARD_MARGIN = 0.04       

SQ_SIZE    = BOARD_SIZE // 8 

FEN_MAP = {
    "white_pawn":   "P", "black_pawn":   "p",
    "white_rook":   "R", "black_rook":   "r",
    "white_knight": "N", "black_knight": "n",
    "white_bishop": "B", "black_bishop": "b",
    "white_queen":  "Q", "black_queen":  "q",
    "white_king":   "K", "black_king":   "k",
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

def _load_yolo_model():
    from ultralytics import YOLO
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    m = YOLO(MODEL_PATH)
    logger.info(f" YOLO Model loaded: {MODEL_PATH}")
    return m

def _load_calibration_data():
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE) as f:
        return json.load(f)

# Global model and cache (lazy load)
_MODEL = None
_CACHE_DATA = None

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame_input, frame_id=0, visualise: bool = True) -> dict:
    global _MODEL, _CACHE_DATA
    
    if _MODEL is None:
        _MODEL = _load_yolo_model()
    if _CACHE_DATA is None:
        _CACHE_DATA = _load_calibration_data()
        
    if _CACHE_DATA is None:
        return _fail("board_cache.json not found. Run aruco_calibration.py first.")

    # 1. Decode Frame
    if isinstance(frame_input, bytes):
        arr = np.frombuffer(frame_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(frame_input, str):
        img = cv2.imread(frame_input)
    else:
        img = frame_input

    if img is None:
        return _fail("Could not read/decode frame.")

    h, w = img.shape[:2]
    
    # 2. Extract and Scale Markers
    # Correct Mapping from aruco_calibration.py:
    # 12: TL, 13: TR, 11: BR, 10: BL
    try:
        centers = _CACHE_DATA["marker_centers"]
        tl = np.array(centers["12"])
        tr = np.array(centers["13"])
        br = np.array(centers["11"])
        bl = np.array(centers["10"])
        
        # Scale if calibration resolution differs from current frame
        if "source_size" in _CACHE_DATA and _CACHE_DATA["source_size"] is not None:
            calib_w, calib_h = _CACHE_DATA["source_size"]
            scale_x = w / calib_w
            scale_y = h / calib_h
            tl = tl * [scale_x, scale_y]
            tr = tr * [scale_x, scale_y]
            br = br * [scale_x, scale_y]
            bl = bl * [scale_x, scale_y]
            
        src_pts = np.float32([tl, tr, br, bl])
        dst_pts = np.float32([[0,0], [BOARD_SIZE,0], [BOARD_SIZE,BOARD_SIZE], [0,BOARD_SIZE]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    except KeyError as e:
        return _fail(f"Missing marker ID in cache: {e}. Re-calibrate.")

    # 3. Detect Pieces on ORIGINAL image
    # Use a medium imgsz for balance between speed and accuracy
    results = _MODEL.predict(img, conf=0.3, iou=0.45, imgsz=640, verbose=False)
    boxes = results[0].boxes
    
    square_detections = {}
    
    for box in boxes:
        cls = int(box.cls[0])
        name = _MODEL.names[cls]
        conf = float(box.conf[0])
        
        if name not in FEN_MAP:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # We use the BOTTOM-CENTER of the box for piece placement (where it touches the board)
        cx = (x1 + x2) / 2
        cy = y2  # Pieces are vertical; the base is at the bottom of the box
        
        # Transform to board space
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_board = cv2.perspectiveTransform(pt, M)[0][0]
        bx, by = pt_board[0], pt_board[1]
        
        # Check if inside board boundaries (with some overflow allowance)
        # We allow -5% to 105% to catch pieces whose base is slightly outside markers
        margin_px = BOARD_SIZE * 0.08
        if -margin_px <= bx < BOARD_SIZE + margin_px and -margin_px <= by < BOARD_SIZE + margin_px:
            
            # Map bx, by to square index [0-7] accounting for the BOARD_MARGIN
            # The playable 8x8 area is within [margin, 1-margin] of the BOARD_SIZE
            inner_w = BOARD_SIZE * (1 - 2 * BOARD_MARGIN)
            inner_h = BOARD_SIZE * (1 - 2 * BOARD_MARGIN)
            
            # Calculate relative position inside the inner 8x8 grid
            rel_x = (bx - BOARD_SIZE * BOARD_MARGIN) / inner_w
            rel_y = (by - BOARD_SIZE * BOARD_MARGIN) / inner_h
            
            col = int(np.clip(rel_x * 8, 0, 7))
            row = int(np.clip(rel_y * 8, 0, 7))
            
            # Highest confidence wins the square
            if (row, col) not in square_detections or conf > square_detections[(row, col)]["conf"]:
                square_detections[(row, col)] = {
                    "char": FEN_MAP[name],
                    "name": name,
                    "conf": conf,
                    "box": (x1, y1, x2, y2),
                    "board_pt": (int(bx), int(by))
                }
        else:
            logger.debug(f"Piece {name} rejected at board coords ({bx:.1f}, {by:.1f})")

    # 4. Generate FEN
    fen_rows = []
    for r in range(8):
        row_str = ""
        empty = 0
        for c in range(8):
            if (r, c) in square_detections:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += square_detections[(r, c)]["char"]
            else:
                empty += 1
        if empty > 0:
            row_str += str(empty)
        fen_rows.append(row_str)
    
    fen = "/".join(fen_rows) + " w - - 0 1"
    
    # 5. Result Summary
    logger.info(f"Frame {frame_id} | FEN: {fen} | Pieces: {len(square_detections)}")
    
    # 6. Step-by-Step Visualization
    if visualise:
        _visualise_results(img, M, square_detections, fen, frame_id)

    return {
        "success": True,
        "fen": fen,
        "pieces_count": len(square_detections),
        "frame_id": frame_id
    }

def _visualise_results(img, M, detections, fen, frame_id):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Generate Warped Board
    warped = cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.canvas.manager.set_window_title(f"FEN Generator - Frame {frame_id}")
    
    # Panel 1: Original + YOLO
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Step 1: YOLO Detection (Frame {frame_id})")
    for (r, c), d in detections.items():
        x1, y1, x2, y2 = d["box"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f"{d['char']} ({d['conf']:.2f})", color='yellow', fontsize=10, weight='bold', bbox=dict(facecolor='black', alpha=0.5))
    axes[0].axis('off')

    # Panel 2: Warped Top-Down
    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Step 2: Correct Perspective Warp")
    # Draw Grid
    for i in range(9):
        axes[1].axhline(i * SQ_SIZE, color='lime', alpha=0.3)
        axes[1].axvline(i * SQ_SIZE, color='lime', alpha=0.3)
    # Draw Piece Centers
    for (r, c), d in detections.items():
        bx, by = d["board_pt"]
        axes[1].plot(bx, by, 'yo', markersize=6)
        axes[1].text(c*SQ_SIZE + 5, r*SQ_SIZE + 20, d["char"], color='white', weight='bold')
    axes[1].axis('off')

    # Panel 3: FEN Generation Result
    axes[2].set_title(f"Step 3: Final FEN Output\n{fen}", fontsize=10)
    axes[2].set_xlim(0, 8)
    axes[2].set_ylim(0, 8)
    for r in range(8):
        for c in range(8):
            color = '#f0d9b5' if (r + c) % 2 == 0 else '#b58863'
            rect = patches.Rectangle((c, 7-r), 1, 1, facecolor=color, edgecolor='black', alpha=0.8)
            axes[2].add_patch(rect)
            if (r, c) in detections:
                char = detections[(r, c)]["char"]
                txt_color = 'black' if char.isupper() else 'white'
                axes[2].text(c + 0.5, 7 - r + 0.5, char, ha='center', va='center', fontsize=20, weight='bold', color=txt_color)
    
    axes[2].set_xticks(np.arange(0.5, 8.5, 1))
    axes[2].set_xticklabels(['a','b','c','d','e','f','g','h'])
    axes[2].set_yticks(np.arange(0.5, 8.5, 1))
    axes[2].set_yticklabels(['1','2','3','4','5','6','7','8'])
    
    plt.tight_layout()
    plt.show()

def _fail(msg):
    logger.error(f"❌ {msg}")
    return {"success": False, "error": msg}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST BLOCK
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    
    # Example usage: python yolo_fen.py path/to/image.jpg
    if len(sys.argv) > 1:
        path = sys.argv[1]
        process_frame(path, frame_id=1, visualise=True)
    else:
        # Check for images folder
        if os.path.exists("images"):
            files = sorted([os.path.join("images", f) for f in os.listdir("images") if f.endswith(('.jpg', '.png'))])
            if files:
                print(f"Processing first image found: {files[0]}")
                process_frame(files[0], frame_id=1, visualise=True)
            else:
                print("No images found in 'images' folder.")
        else:
            print("Usage: python yolo_fen.py <image_path>")