# """
# main.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chess Robot Pipeline — Redis based, No frame_diff

# 3 KEY VARIABLES:
#   yolo_active        → True  = Human turn  = YOLO chalu
#                        False = Robot turn  = YOLO band
#   last_confirmed_fen → Last CONFIRMED board position
#                        (ye Stockfish ko bhi dete hain)
#   candidate_fen      → "Shayad move hua" wali holding jagah
#                        Agar agla YOLO bhi same de → CONFIRM
#                        Agar alag de → reset

# PIPELINE (Human Turn):
#   1. Redis se latest frame lo
#   2. Same frame? Skip
#   3. YOLO → FEN nikalo  [warp_matrix bhi pass hota hai ab]
#   4. CHECK 1: FEN realistic? (kings hain? count sahi?)
#   5. CHECK 2: last_confirmed_fen se alag?
#   6. CHECK 3: 2 baar same FEN aayi? (candidate confirm)
#   7. CHECK 4: move_validator → legal?
#   8. Timer update → Robot turn shuru

# PIPELINE (Robot Turn):
#   yolo_active = False → YOLO band
#   Stockfish → move
#   Robot arm → execute
#   3 sec wait
#   last_confirmed_fen = new FEN
#   yolo_active = True → Human turn shuru

# STARTUP ORDER:
#   Terminal 1: python camera_capture_android.py
#   Terminal 2: python aruco_calibration.py   ← board_cache.json banta hai
#   Terminal 3: python main.py                ← cache load hoti hai yahan

# REDIS KEYS READ:
#   latest_frame      → camera se aya frame (binary)
#   latest_frame_id   → frame number

# REDIS KEYS WRITTEN:
#   board_fen         → current confirmed board position
#   robot_move        → robot ka last UCI move
#   human_hints       → human ke liye suggested moves
#   pipeline_status   → current step (debug ke liye)
# """

# import time
# import threading
# import logging
# import redis

# from yolo_fen           import process_frame
# from move_validator     import MoveValidator
# from chess_brain        import ChessBrain
# from timer              import ChessTimer
# from board_cache_loader import load_board_cache      # ← NEW: ArUco cache loader

# logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
# logger = logging.getLogger(__name__)

# # ── Config ────────────────────────────────────────────────
# REDIS_HOST    = "127.0.0.1"
# REDIS_PORT    = 6379
# GAME_MINUTES  = 10
# DIFFICULTY    = "Medium"
# YOLO_INTERVAL = 3     # har kitne second baad YOLO chale (human turn mein)
# ARM_WAIT_SEC  = 20 # robot arm ke settle hone ka wait

# # ── Starting FEN ──────────────────────────────────────────
# STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# # ══════════════════════════════════════════════════════════
# # PRETTY PRINT HELPERS
# # ══════════════════════════════════════════════════════════

# def sep():
#     print("\n" + "━" * 55)

# def step(n, msg):
#     print(f"\n{'─'*55}")
#     print(f"  STEP {n} │ {msg}")
#     print(f"{'─'*55}")

# def info(msg):      print(f"         ➤  {msg}")
# def ok(msg):        print(f"         ✅ {msg}")
# def warn(msg):      print(f"         ⚠️  {msg}")
# def fail(msg):      print(f"         ❌ {msg}")
# def var(name, val): print(f"         📌 {name} = {val}")


# # ══════════════════════════════════════════════════════════
# # FEN REALISTIC CHECK
# # ══════════════════════════════════════════════════════════

# def count_pieces(fen_position: str) -> dict:
#     """
#     FEN ke sirf pieces wala part (slash se pehle spaces tak) count karo.
#     e.g. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#     """
#     counts = {}
#     for ch in fen_position:
#         if ch.isalpha():
#             counts[ch] = counts.get(ch, 0) + 1
#     return counts


# def is_fen_realistic(prev_fen: str, new_fen: str) -> tuple:
#     """
#     Returns (True/False, reason_string)

#     Yeh check karta hai ke YOLO ne jo FEN di woh physically possible hai ya nahi.
#     Haath/arm beech mein aya → pieces count bilkul galat hoga → FAIL

#     Rules:
#       1. Dono kings hone chahiye (K aur k)
#       2. Koi bhi piece prev se ZYADA nahi ho sakti (pieces kahan se aayi?)
#       3. Ek waqt mein max 1 piece kam ho sakti (sirf capture mein)
#          Zyada kam = haath tha ya obstruction
#     """
#     prev_pos = prev_fen.split(" ")[0]
#     new_pos  = new_fen.split(" ")[0]

#     prev = count_pieces(prev_pos)
#     new  = count_pieces(new_pos)

#     # Rule 1: Kings
#     if new.get('K', 0) != 1:
#         return False, "White King (K) nahi mila — haath tha ya YOLO ka noise"
#     if new.get('k', 0) != 1:
#         return False, "Black King (k) nahi mila — haath tha ya YOLO ka noise"

#     # Rule 2: Koi piece achanak zyada nahi ho sakti
#     for piece, count in new.items():
#         prev_count = prev.get(piece, 0)
#         if count > prev_count + 1:
#             return False, f"Piece '{piece}' achanak {count} ho gayi (pehle {prev_count}) — YOLO galat"

#     # Rule 3: Ek waqt mein max 1 piece hi gayab ho sakti
#     total_prev = sum(prev.values())
#     total_new  = sum(new.values())

#     if total_prev - total_new > 1:
#         return False, f"Bahut saari pieces gayab ({total_prev}→{total_new}) — haath ya obstruction tha"

#     return True, "FEN realistic hai ✅"


# # ══════════════════════════════════════════════════════════
# # FRAME FETCH FROM REDIS
# # ══════════════════════════════════════════════════════════

# def get_latest_frame(r_bin, r_str) -> tuple:
#     """
#     Redis se latest frame bytes aur frame_id lo.
#     Returns (frame_bytes, frame_id) ya (None, None) agar koi frame nahi.
#     """
#     try:
#         frame_bytes = r_bin.get("latest_frame")
#         frame_id    = r_str.get("latest_frame_id")
#         if not frame_bytes:
#             return None, None
#         return frame_bytes, frame_id
#     except Exception as e:
#         logger.error("❌ Redis frame read error: %s", e)
#         return None, None


# # ══════════════════════════════════════════════════════════
# # TIMER DISPLAY THREAD
# # ══════════════════════════════════════════════════════════

# def _fmt(ms):
#     total_sec = max(0, int(ms)) // 1000
#     return f"{total_sec // 60:02d}:{total_sec % 60:02d}"


# def live_timer_display(timer, stop_event):
#     """Background thread — har second timer print karta hai."""
#     while not stop_event.is_set():
#         white_ms, black_ms = timer.get_times()
#         active = timer.active
#         status = timer.status

#         wa = " ◀ RUNNING" if active == "white" and status == "running" else ""
#         ba = " ◀ RUNNING" if active == "black" and status == "running" else ""

#         print(
#             f"\r⚪ White(Robot): {_fmt(white_ms)}{wa:<12}  |  "
#             f"⚫ Black(Human): {_fmt(black_ms)}{ba:<12}  |  "
#             f"{status.upper()}     ",
#             end="", flush=True
#         )

#         if status == "timeout":
#             print(f"\n⏰ TIME'S UP! Winner: {timer.winner.upper()}")
#             break

#         time.sleep(1)


# # ══════════════════════════════════════════════════════════
# # ROBOT TURN
# # ══════════════════════════════════════════════════════════

# def robot_turn(brain, timer, r_str, last_confirmed_fen: str, r_bin) -> str:
#     print("\n" + "═" * 55)
#     print("  🤖 ROBOT'S TURN STARTED")
#     print("═" * 55)

#     # Step R1: Stockfish
#     print("  [1] Stockfish is calculating the best move...")
#     brain_result = brain.process_turn(last_confirmed_fen, difficulty=DIFFICULTY)

#     if not brain_result["success"]:
#         print(f"  ❌ Stockfish fail: {brain_result['error']}")
#         return last_confirmed_fen

#     print(f"  ✅ Best move found: {brain_result['robot_move']}")
    
#     # Redis update
#     r_str.set("robot_move",  brain_result["robot_move"])
#     r_str.set("human_hints", str(brain_result.get("human_hints", [])))

#     # Step R2: Robot arm
#     print(f"  [2] Sending move to Robot Arm...")
#     # execute_move(brain_result["robot_move"])

#     # Step R3: Wait — arm settle ho jaye
#     print(f"  [3] Waiting for arm to finish ({ARM_WAIT_SEC} sec)...")
#     for i in range(ARM_WAIT_SEC, 0, -1):
#         print(f"\r      ⏳ {i} sec...", end="", flush=True)
#         time.sleep(1)
#     print()

#     # Step R4: YOLO (The user wants YOLO to generate the FEN after the move)
#     print("  [4] Running YOLO to verify new board state...")
#     frame_bytes = r_bin.get("latest_frame")
#     if frame_bytes:
#         from yolo_fen import process_frame
#         result = process_frame(frame_bytes, visualise=False)
#         if result["success"]:
#             new_fen = result["fen"]
#             print(f"  ✅ Verified! New FEN: {new_fen}")
#             r_str.set("board_fen", new_fen)
            
#             # Timer: White ruko, Black chalu
#             timer.on_white_move_done()
#             print("\n" + "═" * 55)
#             print("  🧑 HUMAN KI BAARI (Your Turn)")
#             print("═" * 55)
#             return new_fen
            
#     print("  ⚠️ YOLO could not verify. Keeping old FEN.")
#     timer.on_white_move_done()
#     return last_confirmed_fen


# # ══════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════

# def run():

#     # ── Init ──────────────────────────────────────────────
#     validator  = MoveValidator()
#     brain      = ChessBrain()
#     timer      = ChessTimer(minutes=GAME_MINUTES)
#     stop_event = threading.Event()

#     r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
#     r_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

#     # ── Redis init ────────────────────────────────────────
#     r_str.set("board_fen",       STARTING_FEN)
#     r_str.set("robot_move",      "")
#     r_str.set("human_hints",     "")
#     r_str.set("pipeline_status", "starting")

#     # ── NEW: Board Cache Load (ArUco warp matrix) ─────────
#     # aruco_calibration.py pehle chala hona chahiye
#     # strict=True → agar board_cache.json nahi mila → program band
#     warp_matrix, marker_centers = load_board_cache(strict=True)
#     # warp_matrix  → process_frame() ko pass hoga (perspective fix)
#     # marker_centers → future use ke liye (abhi sirf logged)

#     print("\n" + "═" * 55)
#     print("   ♟️   CHESS ROBOT PIPELINE STARTED")
#     print("═" * 55)
#     print(f"   Redis    : {REDIS_HOST}:{REDIS_PORT}")
#     print(f"   Interval : {YOLO_INTERVAL} sec (YOLO)")
#     print(f"   Arm wait : {ARM_WAIT_SEC} sec")
#     print(f"   Time     : {GAME_MINUTES} min per side")
#     print(f"   Markers  : {sorted(marker_centers.keys())}")   # ← NEW: confirm karega
#     print("═" * 55)

#     # ── Timer display thread ──────────────────────────────
#     display_thread = threading.Thread(
#         target=live_timer_display,
#         args=(timer, stop_event),
#         daemon=True
#     )
#     display_thread.start()
#     print("\n[TIMER]  White = Robot  |   Black = Human\n")

#     # ── 3 KEY VARIABLES ───────────────────────────────────
#     yolo_active        = True          # Human ka turn shuru
#     last_confirmed_fen = STARTING_FEN  # Starting position
#     candidate_fen      = ""            # Pehli baar aayi FEN (wait for confirm)
#     last_frame_id      = None          # Same frame skip karne ke liye

#     # ── Robot pehle chalta hai ────────────────────────────
#     # White clock shuru, robot apna pehla move kare
#     timer.start_game()
#     print("\n[TIMER]  White timer SHURU — Robot pehle move karega\n")

#     # ── Robot ka PEHLA move ───────────────────────────────
#     yolo_active = False   # YOLO band — robot arm chalegi

#     var("yolo_active",        yolo_active)
#     var("last_confirmed_fen", last_confirmed_fen)
#     var("candidate_fen",      repr(candidate_fen))

#     last_confirmed_fen = robot_turn(brain, timer, r_str, last_confirmed_fen, r_bin)
#     yolo_active        = True    # Human ki baari
#     candidate_fen      = ""      # Reset

#     print("\n[PIPELINE]  Robot ka pehla move ho gaya — Human ki baari")
#     var("yolo_active",        yolo_active)
#     var("last_confirmed_fen", last_confirmed_fen)
#     var("candidate_fen",      repr(candidate_fen))

#     # ══════════════════════════════════════════════════════
#     # MAIN GAME LOOP
#     # ══════════════════════════════════════════════════════

#     while True:
#         try:

#             # ── Timeout check ─────────────────────────────
#             if timer.status == "timeout":
#                 sep()
#                 print(f"  ⏰ GAME OVER — {timer.winner.upper()} wins on time!")
#                 break

#             # ── YOLO band hai? (robot ka turn) ────────────
#             if not yolo_active:
#                 time.sleep(0.5)
#                 continue

#             # ── Human turn — wait YOLO_INTERVAL ──────────
#             time.sleep(YOLO_INTERVAL)

#             # ══════════════════════════════════════════════
#             # STEP 1 — Redis se frame lo
#             # ══════════════════════════════════════════════
#             r_str.set("pipeline_status", "step1_fetch")
#             frame_bytes, frame_id = get_latest_frame(r_bin, r_str)

#             if frame_bytes is None:
#                 continue


#             # ══════════════════════════════════════════════
#             # STEP 2 — Same frame? Skip
#             # ══════════════════════════════════════════════
#             if frame_id == last_frame_id:
#                 continue

#             last_frame_id = frame_id
            
#             # Simple print to let the user know what's happening
#             print(f"\n[PIPELINE] Frame {frame_id} received. Running YOLO and checking for human move...")

#             # ══════════════════════════════════════════════
#             # STEP 3 — YOLO  [warp_matrix ab pass ho raha hai]
#             # ══════════════════════════════════════════════
#             r_str.set("pipeline_status", "step3_yolo")

#             result = process_frame(frame_bytes, visualise=False)

#             if not result["success"]:
#                 print("         ⚠️ YOLO failed to process frame.")
#                 continue

#             new_fen = result["fen"]
#             print(f"         ➤ Pieces detected: {result['pieces_count']} | FEN: {new_fen}")

#             # # ══════════════════════════════════════════════
#             # # CHECK 1 — FEN realistic hai?
#             # # Kings hain? Pieces count sahi?
#             # # ══════════════════════════════════════════════
#             # step(4, "CHECK 1 — FEN realistic hai? (Kings? Count sahi?)")
#             # r_str.set("pipeline_status", "step4_check1")

#             # realistic, reason = is_fen_realistic(last_confirmed_fen, new_fen)

#             # if not realistic:
#             #     fail(f"FEN reject: {reason}")
#             #     info("Haath tha ya arm tha ya YOLO noise — skip")
#             #     candidate_fen = ""   # reset
#             #     var("candidate_fen", repr(candidate_fen))
#             #     continue

#             # ok(f"FEN realistic: {reason}")

#             # ══════════════════════════════════════════════
#             # CHECK 2 — last_confirmed_fen se alag hai?
#             # ══════════════════════════════════════════════
#             r_str.set("pipeline_status", "step5_check2")

#             new_pos  = new_fen.split(" ")[0]
#             prev_pos = last_confirmed_fen.split(" ")[0]

#             if new_pos == prev_pos:
#                 print("         ➤ Board unchanged. Waiting...")
#                 candidate_fen = ""   # reset
#                 continue

#             ok("FEN alag hai — kuch badla!")

#             # # ══════════════════════════════════════════════
#             # # CHECK 3 — 2 baar same FEN aayi? (Confirm)
#             # # ══════════════════════════════════════════════
#             # step(6, "CHECK 3 — 2 consecutive frames mein same FEN? (confirm check)")
#             # r_str.set("pipeline_status", "step6_check3")

#             # if candidate_fen == "":
#             #     candidate_fen = new_pos
#             #     info("Pehli baar alag FEN aayi → candidate mein save kiya")
#             #     info("Ek aur frame ka wait — confirm karna hai")
#             #     var("candidate_fen", candidate_fen[:40] + "...")
#             #     continue

#             # if new_pos != candidate_fen:
#             #     warn("FEN shift ho gayi — abhi settle nahi hua, naya candidate set")
#             #     candidate_fen = new_pos
#             #     var("candidate_fen", candidate_fen[:40] + "...")
#             #     continue

#             ok("FEN alag aayi — MOVE CONFIRM! 🎉")
#             candidate_fen = ""   # reset

#             # ══════════════════════════════════════════════
#             # CHECK 4 — move_validator (TEMP DISABLED)
#             # ══════════════════════════════════════════════

#             # step(7, "CHECK 4 — Chess rules check (move_validator)...")
#             # r_str.set("pipeline_status", "step7_validate")

#             # info(f"FEN before : {last_confirmed_fen}")
#             # info(f"FEN after  : {new_fen}")

#             # is_legal, move_uci, reason = validator.validate_and_commit(
#             #     last_confirmed_fen, new_fen
#             # )

#             # if not is_legal:
#             #     fail(f"Illegal move! Reason: {reason}")
#             #     warn("Sahi move karo — dobara wait kar raha hoon")
#             #     candidate_fen = ""
#             #     continue

#             # ok(f"Move LEGAL! UCI: {move_uci}")

#             # ══════════════════════════════════════════════
#             # HUMAN MOVE ACCEPTED (NO VALIDATION MODE)
#             # ══════════════════════════════════════════════
#             last_confirmed_fen = new_fen
#             candidate_fen      = ""

#             r_str.set("board_fen",        last_confirmed_fen)
#             r_str.set("pipeline_status",  "human_move_done")

#             ok("Move accept kar liya (validation OFF)")

#             # Timer: Black ruko, White chalu
#             timer.on_black_move_detected()
#             ok("⚫ Black timer RUKA")
#             ok("⚪ White timer CHALU — Robot ki baari")

#             var("yolo_active",        "False (robot turn)")
#             var("last_confirmed_fen", last_confirmed_fen)
#             var("candidate_fen",      repr(candidate_fen))

#             # ══════════════════════════════════════════════
#             # ROBOT TURN
#             # ══════════════════════════════════════════════

#             yolo_active = False   # YOLO band karo — robot chalega

#             last_confirmed_fen = robot_turn(
#                 brain, timer, r_str, last_confirmed_fen, r_bin
#             )

#             yolo_active   = True   # Human ki baari dobara
#             candidate_fen = ""     # Reset

#             r_str.set("pipeline_status", "waiting_human")

#         except KeyboardInterrupt:
#             sep()
#             print("  🛑 Ctrl+C — Band ho raha hoon")
#             stop_event.set()
#             timer.stop()
#             brain.close()
#             break
#         except Exception as e:
#             logger.error("⚠️  Unexpected error: %s", e, exc_info=True)
#             time.sleep(2)


# if __name__ == "__main__":
#     run()

"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess Robot Pipeline — Redis based, No frame_diff
"""

import time
import threading
import logging
import redis

from yolo_fen           import process_frame
from move_validator     import MoveValidator
from chess_brain        import ChessBrain
from timer              import ChessTimer
from board_cache_loader import load_board_cache

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
ARM_WAIT_SEC  = 10

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
        return False, "White King (K) nahi mila"
    if new.get('k', 0) != 1:
        return False, "Black King (k) nahi mila"

    for piece, count in new.items():
        prev_count = prev.get(piece, 0)
        if count > prev_count + 1:
            return False, f"Piece '{piece}' achanak {count} ho gayi"

    total_prev = sum(prev.values())
    total_new  = sum(new.values())

    if total_prev - total_new > 1:
        return False, f"Bahut saari pieces gayab ({total_prev}→{total_new})"

    return True, "FEN realistic hai ✅"


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
    print("  🤖 ROBOT'S TURN STARTED")
    print("═" * 55)

    print(f"  [1] Stockfish is calculating the best move for FEN:\n      ➤ {last_confirmed_fen}")
    brain_result = brain.process_turn(last_confirmed_fen, difficulty=DIFFICULTY)

    if not brain_result["success"]:
        print(f"  ❌ Stockfish fail: {brain_result['error']}")
        return last_confirmed_fen

    print(f"  ✅ Best move found: {brain_result['robot_move']}")
    
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
            print(f"  ✅ Verified! New FEN: {new_fen}")
            r_str.set("board_fen", new_fen)
            
            timer.on_white_move_done()
            print("\n" + "═" * 55)
            print("  🧑 HUMAN KI BAARI (Your Turn)")
            print("═" * 55)
            return new_fen
            
    print("  ⚠️ YOLO could not verify. Keeping old FEN.")
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
    print("   ♟️   CHESS ROBOT PIPELINE STARTED")
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
                print(f"\n  ⏰ GAME OVER — {timer.winner.upper()} wins on time!")
                break

            if not yolo_active:
                time.sleep(0.5)
                continue

            # ── Human turn — wait YOLO_INTERVAL ──────────
            print(f"\n      ⏳ Waiting {YOLO_INTERVAL} seconds for next YOLO check...")
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
                print("         ⚠️ YOLO failed to process frame.")
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
            print("         ➤ 📊 Opening visualization window (Close the window to continue)...")
            
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
            print("\n  🛑 Ctrl+C — Band ho raha hoon")
            stop_event.set()
            timer.stop()
            brain.close()
            break
        except Exception as e:
            logger.error("⚠️  Unexpected error: %s", e, exc_info=True)
            time.sleep(2)

if __name__ == "__main__":
    run()
