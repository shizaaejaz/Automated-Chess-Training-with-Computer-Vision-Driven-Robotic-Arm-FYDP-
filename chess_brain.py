import chess
import chess.engine
import logging
import redis

logger = logging.getLogger(__name__)

STOCKFISH_PATH = r"C:\Users\shiza\OneDrive\Desktop\sw_making\stockfish\stockfish-windows-x86-64-avx2.exe"

# Redis connection 
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379


def get_redis():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        socket_connect_timeout=5,
        decode_responses=True,
    )
    r.ping()
    return r

# FEN SANITIZER HELPERS

def _expand_row(row: str) -> list:
    """
    Expand a FEN row string into a list of 1 character per square.
    Empty squares become "."

    Example:
        "3P2"  → ['.', '.', '.', 'P', '.', '.']
        "PPPP1PP" → ['P','P','P','P','.','P','P']
    """
    result = []
    for ch in row:
        if ch.isdigit():
            result.extend(["."] * int(ch))
        else:
            result.append(ch)
    return result


def _compress_row(cols: list) -> str:
    """
    Compress a list of squares back into a FEN row string.
    "." becomes empty count digit.

    Example:
        ['.', '.', 'P', '.'] → "2P1"
        ['P','P','P','P','.','P','P'] → "PPPP1PP"
    """
    result = ""
    empty  = 0
    for ch in cols:
        if ch == ".":
            empty += 1
        else:
            if empty:
                result += str(empty)
                empty = 0
            result += ch
    if empty:
        result += str(empty)
    return result


def _fix_row_width(row: str) -> str:
    """
    Ensures a FEN row represents exactly 8 squares.
    Trims if too long, pads with empty squares if too short.

    This fixes the main crash:
      "expected 8 columns per row in position part of fen"
    """
    cols = _expand_row(row)
    if len(cols) > 8:
        logger.warning("[ChessBrain] Row '%s' has %d cols — trimming to 8", row, len(cols))
        cols = cols[:8]
    elif len(cols) < 8:
        logger.warning("[ChessBrain] Row '%s' has %d cols — padding to 8", row, len(cols))
        cols = cols + ["."] * (8 - len(cols))
    return _compress_row(cols)


def _keep_first_piece(rows: list, piece: str) -> list:
    """
    If a piece character appears more than once across all rows,
    keep only the first occurrence and replace extras with empty square "1".

    Works on the expanded/raw row strings (not yet compressed).
    """
    found = False
    new_rows = []
    for row in rows:
        new_row = ""
        for ch in row:
            if ch == piece:
                if not found:
                    new_row += ch   # keep first
                    found = True
                else:
                    new_row += "1"  # replace extra with 1 empty square
                    logger.info("[ChessBrain] Extra '%s' removed from FEN", piece)
            else:
                new_row += ch
        # Re-compress the row so adjacent empty squares (like '1' + '2') are correctly combined into '3'
        new_rows.append(_compress_row(_expand_row(new_row)))
    return new_rows


def _place_piece_at(rows: list, piece: str, square: str) -> list:
    """
    Place a piece at a specific square in the FEN rows list.
    Uses proper row expansion so the column count stays exactly 8.

    square: chess notation e.g. "e1" or "e8"
    """
    file_idx = ord(square[0]) - ord('a')   # 'e' → 4
    rank     = int(square[1])              # 1 or 8
    row_idx  = 8 - rank                    # rank 8 → row 0, rank 1 → row 7

    cols = _expand_row(rows[row_idx])

    # Pad/trim to 8 just in case
    while len(cols) < 8:
        cols.append(".")
    cols = cols[:8]

    cols[file_idx] = piece
    rows[row_idx]  = _compress_row(cols)
    return rows


def _sanitize_fen(fen: str) -> str:

    try:
        parts      = fen.strip().split()
        piece_part = parts[0]
        rows       = piece_part.split("/")

        # Need exactly 8 rows
        while len(rows) < 8:
            rows.append("8")
        rows = rows[:8]

        # Step 1: Fix every row to exactly 8 columns 
        rows = [_fix_row_width(r) for r in rows]

        # Step 2: Remove duplicate white King 
        flat = "".join(rows)
        if flat.count("K") > 1:
            rows = _keep_first_piece(rows, "K")

        # Step 3: Remove duplicate black king 
        flat = "".join(rows)
        if flat.count("k") > 1:
            rows = _keep_first_piece(rows, "k")

        #Step 4: Remove duplicate white Queen 
        flat = "".join(rows)
        if flat.count("Q") > 1:
            rows = _keep_first_piece(rows, "Q")

        # Step 5: Remove duplicate black queen 
        flat = "".join(rows)
        if flat.count("q") > 1:
            rows = _keep_first_piece(rows, "q")

        # Step 6: Add missing white King 
        flat = "".join(rows)
        if flat.count("K") == 0:
            logger.info("[ChessBrain] No white King — placing at e1")
            rows = _place_piece_at(rows, "K", "e1")

        # Step 7: Add missing black king 
        flat = "".join(rows)
        if flat.count("k") == 0:
            logger.info("[ChessBrain] No black King — placing at e8")
            rows = _place_piece_at(rows, "k", "e8")

        # Step 8: Rebuild FEN 
        fixed_piece_part = "/".join(rows)
        fixed_fen        = fixed_piece_part + " w - - 0 1"

        #Step 9: Validate with python-chess 
        chess.Board(fixed_fen)   # throws ValueError if still broken
        logger.info("[ChessBrain] FEN after sanitize: %s", fixed_fen)
        return fixed_fen

    except Exception as e:
        logger.error("[ChessBrain] Sanitize failed: %s", e)
        return None

# CHESS BRAIN CLASS


class ChessBrain:

    def __init__(self, stockfish_path=STOCKFISH_PATH):
        self.path  = stockfish_path
        self.redis = get_redis()
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            logger.info("[ChessBrain] ✅ Stockfish connected")
        except Exception as e:
            logger.error("[ChessBrain] ❌ Stockfish connection failed: %s", e)
            self.engine = None

    def process_turn(self, current_fen: str, difficulty: str = "Medium") -> dict:

        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            except Exception as e:
                return _fail(f"Could not reconnect Stockfish: {e}")

        if not current_fen:
            return _fail("No FEN received")

        # sanitize
        sanitized = _sanitize_fen(current_fen)
        if sanitized is None:
            return _fail("FEN sanitize failed — skipping turn")

        current_fen = sanitized

        try:
            board = chess.Board(current_fen)
        except ValueError as e:
            return _fail(f"Invalid FEN: {e}")

        # difficulty
        skill_map = {"Easy": 5, "Medium": 12, "Hard": 20}
        self.engine.configure({"Skill Level": skill_map.get(difficulty, 12)})

        # robot move
        try:
            result     = self.engine.play(board, chess.engine.Limit(time=0.5))
            robot_move = result.move
        except Exception as e:
            self.engine = None
            return _fail(f"Stockfish crashed: {e}")

        # apply move internally (no FEN return)
        board.push(robot_move)

        # hints
        try:
            analysis = self.engine.analyse(
                board, chess.engine.Limit(time=0.5), multipv=3
            )
            human_hints = [entry["pv"][0].uci() for entry in analysis]
        except Exception:
            human_hints = []

        # Redis (NO FEN)
        try:
            pipe = self.redis.pipeline()
            pipe.set("robot_move", robot_move.uci())
            pipe.set("human_hints", str(human_hints))
            pipe.execute()
        except Exception:
            pass

        return {
            "success":     True,
            "error":       None,
            "robot_move":  robot_move.uci(),
            "new_fen":     board.fen(),
            "human_hints": human_hints,
            "difficulty":  difficulty,
        }

    def close(self):
        if self.engine:
            self.engine.quit()


# ── FAIL HELPER (UPDATED) ──
def _fail(msg: str) -> dict:
    return {
        "success":     False,
        "error":       msg,
        "robot_move":  None,
        "human_hints": [],
        "difficulty":  None,
    }



if __name__ == "__main__":
    brain = ChessBrain()

    # Starting chess position
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

    result = brain.process_turn(test_fen, "Medium")

    print("\n=== RESULT ===")
    print("Success:", result["success"])
    print("Move:", result["robot_move"])
    print("Hints:", result["human_hints"])
    print("Difficulty:", result["difficulty"])
    print("Error:", result["error"])

    brain.close()