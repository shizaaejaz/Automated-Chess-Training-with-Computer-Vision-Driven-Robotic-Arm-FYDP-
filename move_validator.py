"""
move_validator.py
━━━━━━━━━━━━━━━━━
Given the OLD board FEN (before a move) and NEW board FEN (after a move,
detected by the vision pipeline), determines:

  1. Whether the position actually changed at all.
  2. Which piece moved and which UCI move string it corresponds to.
  3. Whether that move is LEGAL according to the chess rules.
"""

import chess
import logging
import redis
from typing import Optional

logger = logging.getLogger(__name__)

# ── Redis connection ──────────────────────────────────────────────────────────
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
# NO PASSWORD!

def get_redis() -> redis.Redis:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        socket_connect_timeout=5,
        decode_responses=True,
    )
    r.ping()
    return r


# ── Pure validation logic 

def _normalise_fen(fen: str) -> str:
    """
    Sirf piece placement part lo FEN se.
    Turn, castling, en-passant, counters sab ignore karo.
    Kyunki YOLOv11 sirf pieces dekh sakta hai!

    Example:
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        becomes:
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    """
    parts = fen.strip().split()
    return parts[0]   # SIRF pieces wala part!


def extract_move_uci(old_fen: str, new_fen: str) -> Optional[str]:
    """
    Dono FEN strings se pata karo kaunsa move hua.
    Har legal move try karta hai aur match dhundta hai.
    """
    try:
        old_board = chess.Board(old_fen)
    except ValueError as e:
        logger.error("[MoveVal] Invalid old FEN '%s': %s", old_fen, e)
        return None

    new_fen_normalised = _normalise_fen(new_fen)

    for move in old_board.legal_moves:
        test_board = old_board.copy()
        test_board.push(move)
        if _normalise_fen(test_board.fen()) == new_fen_normalised:
            return move.uci()

    return None


def is_valid_move(old_fen: str, new_fen: str) -> tuple:
    """
    Main validation function.

    Returns: (is_legal, move_uci, reason)
    """
    # FEN valid hai?
    for label, fen in [("old", old_fen), ("new", new_fen)]:
        try:
            chess.Board(fen)
        except ValueError as e:
            msg = f"Invalid {label} FEN: {e}"
            logger.warning("[MoveVal] %s", msg)
            return False, None, msg

    # Kuch badla bhi?
    if _normalise_fen(old_fen) == _normalise_fen(new_fen):
        return False, None, "Board position unchanged – no move detected."

    # Legal move dhundo
    move_uci = extract_move_uci(old_fen, new_fen)

    if move_uci:
        logger.info("[MoveVal] ✅ Legal move detected: %s", move_uci)
        return True, move_uci, f"Legal move: {move_uci}"
    else:
        msg = "No legal move matches the detected board change. Possibly illegal or mid-move."
        logger.warning("[MoveVal] ❌ %s", msg)
        return False, None, msg


# ── Redis-integrated validator class ─────────────────────────────────────────

class MoveValidator:

    def __init__(self):
        self.redis = get_redis()
        logger.info("[MoveVal] Initialised. Connected to Redis %s:%d", REDIS_HOST, REDIS_PORT)

    def validate_and_commit(self, old_fen: str, new_fen: str) -> tuple:
        is_legal, move_uci, reason = is_valid_move(old_fen, new_fen)
        self._write_result(is_legal, move_uci, reason)
        return is_legal, move_uci, reason

    def get_current_fen(self) -> Optional[str]:
        return self._get_current_fen()

    def _get_current_fen(self) -> Optional[str]:
        try:
            return self.redis.get("board_fen")
        except Exception as e:
            logger.error("[MoveVal] Redis GET board_fen failed: %s", e)
            return None

    def _write_result(self, is_legal: bool, move_uci: Optional[str], reason: str):
        try:
            pipe = self.redis.pipeline()
            pipe.set("move_valid", str(is_legal))
            pipe.set("move_error", reason if not is_legal else "")
            if is_legal and move_uci:
                pipe.set("validated_move", move_uci)
                pipe.set("last_move",      move_uci)
            pipe.execute()
        except Exception as e:
            logger.error("[MoveVal] Redis write failed: %s", e)