

"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess Robot Pipeline — Redis based, No frame_diff
"""

import time
import threading
import logging
import redis
import serial

from yolo_fen           import process_frame
from move_validator     import MoveValidator
from chess_brain        import ChessBrain
from timer              import ChessTimer
from board_cache_loader import load_board_cache
from feedback           import announce_best_move

# Open a port (e.g., COM3 on Windows or /dev/ttyUSB0 on Linux)
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(5)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Silence other modules so they don't spam INFO logs
logging.getLogger("yolo_fen").setLevel(logging.WARNING)
logging.getLogger("chess_brain").setLevel(logging.WARNING)
logging.getLogger("move_validator").setLevel(logging.WARNING)
logging.getLogger("timer").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────
REDIS_HOST    = "127.0.0.1"
REDIS_PORT    = 6379
GAME_MINUTES  = 10
DIFFICULTY    = "Medium"
YOLO_INTERVAL = 3     
ARM_WAIT_SEC  = 50

# ── Starting FEN ──────────────────────────────────────────
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def count_pieces(fen_position: str) -> dict:
    counts = {}
    for ch in fen_position:
        if ch.isalpha():
            counts[ch] = counts.get(ch, 0) + 1
    return counts

def is_fen_realistic(prev_fen: str, new_fen: str) -> tuple:
    prev_pos = prev_fen.split(" ")[0]
    new_pos  = new_fen.split(" ")[0]

    prev = count_pieces(prev_pos)
    new  = count_pieces(new_pos)

    if new.get('K', 0) != 1:
        return False, "White King (K) not found"
    if new.get('k', 0) != 1:
        return False, "Black King (k) not found"

    for piece, count in new.items():
        prev_count = prev.get(piece, 0)
        if count > prev_count + 1:
            return False, f"Piece '{piece}' achanak {count} ho gayi"

    total_prev = sum(prev.values())
    total_new  = sum(new.values())

    if total_prev - total_new > 1:
        return False, f"Too many missing pieces ({total_prev}→{total_new})"

    return True, "FEN is realistic"


def get_latest_frame(r_bin, r_str) -> tuple:
    try:
        frame_bytes = r_bin.get("latest_frame")
        frame_id    = r_str.get("latest_frame_id")
        if not frame_bytes:
            return None, None
        return frame_bytes, frame_id
    except Exception as e:
        logger.error("❌ Redis frame read error: %s", e)
        return None, None


def _fmt(ms):
    total_sec = max(0, int(ms)) // 1000
    return f"{total_sec // 60:02d}:{total_sec % 60:02d}"


def live_timer_display(timer, stop_event):
    while not stop_event.is_set():
        white_ms, black_ms = timer.get_times()
        active = timer.active
        status = timer.status

        wa = " ◀ RUNNING" if active == "white" and status == "running" else ""
        ba = " ◀ RUNNING" if active == "black" and status == "running" else ""

        print(
            f"\r⚪ White(Robot): {_fmt(white_ms)}{wa:<12}  |  "
            f"⚫ Black(Human): {_fmt(black_ms)}{ba:<12}  |  "
            f"{status.upper()}     ",
            end="", flush=True
        )

        if status == "timeout":
            print(f"\n⏰ TIME'S UP! Winner: {timer.winner.upper()}")
            break

        time.sleep(1)


def robot_turn(brain, timer, r_str, last_confirmed_fen: str, r_bin) -> str:
    print("\n" + "═" * 55)
    print("   ROBOT'S TURN STARTED")
    print("═" * 55)

    print(f"  [1] Stockfish is calculating the best move for FEN:\n      ➤ {last_confirmed_fen}")
    brain_result = brain.process_turn(last_confirmed_fen, difficulty=DIFFICULTY)

    if not brain_result["success"]:
        print(f"   Stockfish fail: {brain_result['error']}")
        return last_confirmed_fen

    print(f"   Best move found: {brain_result['robot_move']}")

    # Announce the move via speaker (non-blocking)
    try:
        announce_best_move(brain_result['robot_move'], blocking=False)
    except Exception:
        # Don't let TTS failures stop the robot pipeline
        pass

    # The data you want to send
    move = brain_result['robot_move']
    # Format the string, add newline for readStringUntil('\n')
    data_to_send = f"{move}&" 

    # 2. Convert to bytes and send
    ser.write(data_to_send.encode('utf-8'))
    print(f"Sent: {data_to_send}")

    #ser.close()
    
    r_str.set("robot_move",  brain_result["robot_move"])
    r_str.set("human_hints", str(brain_result.get("human_hints", [])))

    print(f"  [2] Sending move to Robot Arm...")

    print(f"  [3] Waiting for arm to finish ({ARM_WAIT_SEC} sec)...")
    for i in range(ARM_WAIT_SEC, 0, -1):
        print(f"\r      ⏳ {i} sec...", end="", flush=True)
        time.sleep(1)
    print()

    print("  [4] Running YOLO to verify new board state...")
    frame_bytes = r_bin.get("latest_frame")
    if frame_bytes:
        from yolo_fen import process_frame
        result = process_frame(frame_bytes, frame_id=r_str.get("latest_frame_id"), visualise=False)
        if result["success"]:
            new_fen = result["fen"]
            print(f"   Verified! New FEN: {new_fen}")
            r_str.set("board_fen", new_fen)
            
            timer.on_white_move_done()
            print("\n" + "═" * 55)
            print("   HUMAN'S TURN ")
            print("═" * 55)
            return new_fen
            
    print("YOLO could not verify. Keeping old FEN.")
    timer.on_white_move_done()
    return last_confirmed_fen


def run():
    validator  = MoveValidator()
    brain      = ChessBrain()
    timer      = ChessTimer(minutes=GAME_MINUTES)
    stop_event = threading.Event()

    r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    r_str.set("board_fen",       STARTING_FEN)
    r_str.set("robot_move",      "")
    r_str.set("human_hints",     "")
    r_str.set("pipeline_status", "starting")

    warp_matrix, marker_centers = load_board_cache(strict=True)

    print("\n" + "═" * 55)
    print("    CHESS ROBOT PIPELINE STARTED")
    print("═" * 55)
    print(f"   Redis    : {REDIS_HOST}:{REDIS_PORT}")
    print(f"   Interval : {YOLO_INTERVAL} sec (YOLO)")
    print(f"   Arm wait : {ARM_WAIT_SEC} sec")
    print(f"   Time     : {GAME_MINUTES} min per side")
    print(f"   Markers  : {sorted(marker_centers.keys())}")
    print("═" * 55)
    print()

    display_thread = threading.Thread(
        target=live_timer_display,
        args=(timer, stop_event),
        daemon=True
    )
    display_thread.start()

    yolo_active        = True
    last_confirmed_fen = STARTING_FEN
    candidate_fen      = ""
    last_frame_id      = None

    timer.start_game()
    yolo_active = False

    last_confirmed_fen = robot_turn(brain, timer, r_str, last_confirmed_fen, r_bin)
    yolo_active        = True
    candidate_fen      = ""

    while True:
        try:
            if timer.status == "timeout":
                print(f"\n   GAME OVER — {timer.winner.upper()} wins on time!")
                break

            if not yolo_active:
                time.sleep(0.5)
                continue

            # ── Human turn — wait YOLO_INTERVAL ──────────
            print(f"\n       Waiting {YOLO_INTERVAL} seconds for next YOLO check...")
            time.sleep(YOLO_INTERVAL)

            r_str.set("pipeline_status", "step1_fetch")
            frame_bytes, frame_id = get_latest_frame(r_bin, r_str)

            if frame_bytes is None:
                continue

            if frame_id == last_frame_id:
                continue

            last_frame_id = frame_id
            current_time = time.strftime('%H:%M:%S')
            print(f"\n[PIPELINE] Frame {frame_id} received at {current_time}. Running YOLO to check for human move...")

            r_str.set("pipeline_status", "step3_yolo")
            result = process_frame(frame_bytes, frame_id=frame_id, visualise=False)

            if not result["success"]:
                print(" YOLO failed to process frame.")
                continue

            new_fen = result["fen"]
            print(f"         ➤ Pieces detected: {result['pieces_count']} | FEN: {new_fen}")

            r_str.set("pipeline_status", "step5_check2")

            new_pos  = new_fen.split(" ")[0]
            prev_pos = last_confirmed_fen.split(" ")[0]

            print(f"         ➤ Comparing FENs:")
            print(f"            - Previous: {prev_pos}")
            print(f"            - New YOLO: {new_pos}")

            if new_pos == prev_pos:
                print("         ➤ Difference: None (Board unchanged). Waiting...")
                candidate_fen = ""
                continue

            print("         ➤ Difference found! Human move accepted.")
            print("         ➤ Opening visualization window (Close the window to continue)...")
            
            # Show the plot
            process_frame(frame_bytes, frame_id=frame_id, visualise=True)

            last_confirmed_fen = new_fen
            candidate_fen      = ""

            r_str.set("board_fen",        last_confirmed_fen)
            r_str.set("pipeline_status",  "human_move_done")

            timer.on_black_move_detected()

            yolo_active = False

            last_confirmed_fen = robot_turn(
                brain, timer, r_str, last_confirmed_fen, r_bin
            )

            yolo_active   = True
            candidate_fen = ""

            r_str.set("pipeline_status", "waiting_human")

        except KeyboardInterrupt:
            print("\n Ctrl+C — Exiting")
            stop_event.set()
            timer.stop()
            brain.close()
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            time.sleep(2)

if __name__ == "__main__":
    run()
