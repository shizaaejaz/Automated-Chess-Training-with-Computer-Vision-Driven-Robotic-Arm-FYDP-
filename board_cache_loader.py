"""
board_cache_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Helper module — main.py imports this.

Loads board_cache.json (written by aruco_calibration.py)
and returns the pre-computed warp matrix + marker centers.

USAGE in main.py:
─────────────────
    from board_cache_loader import load_board_cache

    warp_matrix, marker_centers = load_board_cache()
    # Pass warp_matrix to yolo_fen.process_frame(frame, warp_matrix=warp_matrix)
"""

import json
import numpy as np
import os
import sys

CACHE_FILE = "board_cache.json"

def load_board_cache(strict=True):
    """
    Loads the ArUco calibration cache.

    Args:
        strict (bool): If True, exit the program if cache is missing.
                       If False, return (None, None) instead.

    Returns:
        (warp_matrix: np.ndarray, marker_centers: dict)
        warp_matrix  → 3×3 float32 perspective matrix
        marker_centers → {int marker_id: np.array([x, y])} for IDs 10-13
    """
    if not os.path.exists(CACHE_FILE):
        msg = (
            f"\n❌  '{CACHE_FILE}' not found!\n"
            f"    Run aruco_calibration.py FIRST to detect board corners:\n"
            f"\n"
            f"    Terminal 1: python camera_capture_android.py\n"
            f"    Terminal 2: python aruco_calibration.py\n"
            f"    Terminal 3: python main.py  ← (you are here)\n"
        )
        if strict:
            print(msg)
            sys.exit(1)
        else:
            print(msg)
            return None, None

    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)

        # Reconstruct warp matrix
        warp_matrix = np.array(data["warp_matrix"], dtype=np.float32)

        # Reconstruct marker centers as numpy arrays
        marker_centers = {
            int(mid): np.array(xy, dtype=np.float32)
            for mid, xy in data["marker_centers"].items()
        }

        print(f"✅ Board cache loaded from '{CACHE_FILE}'")
        print(f"   Calibrated at: {data.get('timestamp', 'unknown')}")
        print(f"   Markers loaded: {sorted(marker_centers.keys())}")
        print()

        return warp_matrix, marker_centers

    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        if strict:
            sys.exit(1)
        return None, None


def get_cached_warp_matrix():
    """Convenience function — returns only the warp matrix (3×3 np.float32)."""
    warp_matrix, _ = load_board_cache()
    return warp_matrix