"""
timer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess Clock — CV-Driven Robotic Arm Project

WHO PLAYS WHAT:
  ♔ White  →  Robotic Arm  (Stockfish moves)
  ♚ Black  →  Human Player (camera se detect hota hai)

CLOCK SWITCHING:
  Game start           → White clock runs (robot pehle)
  Robot arm done       → White STOPS, Black STARTS
  Human move confirmed → Black STOPS, White STARTS
  Either hits 0:00     → Game over

REDIS KEYS WRITTEN (har second):
  timer:white_ms    → White time left (milliseconds)
  timer:black_ms    → Black time left (milliseconds)
  timer:white_fmt   → "09:45" (human readable)
  timer:black_fmt   → "09:52"
  timer:active      → "white" | "black" | "stopped"
  timer:status      → "running" | "paused" | "timeout"
  timer:winner      → "white" | "black" | ""
  timer:white_moves → move count
  timer:black_moves → move count

WHAT main.py CALLS:
  timer = ChessTimer(minutes=10)
  timer.start_game()             ← game shuru, white clock start
  timer.on_white_move_done()     ← robot arm done ke baad
  timer.on_black_move_detected() ← human valid move confirm ke baad
  timer.stop()                   ← game end pe cleanup
  white_ms, black_ms = timer.get_times()
"""

import time
import threading
import logging
import os
import redis

logger = logging.getLogger(__name__)

REDIS_HOST      = "127.0.0.1"
REDIS_PORT      = 6379
DEFAULT_MINUTES = 10

TIMER_LOG_FILE  = "timer_log.txt"   # GUI friend reads this


# ══════════════════════════════════════════════════════════
# CHESS TIMER
# ══════════════════════════════════════════════════════════

class ChessTimer:
    """
    Thread-safe chess clock using wall-clock time (time.time()).

    Background thread har second:
      - Active player ka time deduct karta hai
      - Timeout check karta hai
      - Redis mein save karta hai
      - timer_log.txt mein save karta hai (GUI ke liye)
    """

    def __init__(self, minutes: int = DEFAULT_MINUTES):
        total_ms = minutes * 60 * 1000

        self.white_ms    = total_ms
        self.black_ms    = total_ms

        self.active      = "stopped"   # "white" | "black" | "stopped"
        self.status      = "paused"    # "running" | "paused" | "timeout"
        self.winner      = ""

        self.white_moves = 0
        self.black_moves = 0

        self._tick_start = None
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()

        self.redis = self._connect_redis()

        os.makedirs(os.path.dirname(os.path.abspath(TIMER_LOG_FILE)), exist_ok=True)

        self._thread = threading.Thread(
            target=self._tick_loop,
            daemon=True,
            name="ChessTimerThread"
        )
        self._thread.start()

        logger.info("[Timer] ✅ Initialised — %d min per side", minutes)

    # ── PUBLIC API ────────────────────────────────────────

    def start_game(self):
        """
        Game shuru — White (robot) clock start.
        ONCE call karo game start pe.
        """
        with self._lock:
            self.active      = "white"
            self.status      = "running"
            self._tick_start = time.time()
        self._push_redis()
        self._save_to_file()

        print("\n" + "=" * 50)
        print("  ♟️  CHESS GAME STARTED")
        print(f"  ⚪ White (Robot) : {_fmt(self.white_ms)}")
        print(f"  ⚫ Black (Human) : {_fmt(self.black_ms)}")
        print("  ⚪ White clock running first (Robot moves first)")
        print("=" * 50 + "\n")

    def on_white_move_done(self):
        """
        Robot arm ka move complete hua.
        White STOPS → Black STARTS.
        main.py mein robot turn ke end pe call karo.
        """
        with self._lock:
            if self.status == "timeout":
                return
            self._flush()
            self.white_moves += 1
            self.active       = "black"
            self.status       = "running"
            self._tick_start  = time.time()
        self._push_redis()
        self._save_to_file()
        logger.info(
            "[Timer] ⚪→⚫  White move #%d | White: %s | Black: %s",
            self.white_moves, _fmt(self.white_ms), _fmt(self.black_ms)
        )

    def on_black_move_detected(self):
        """
        Human ka valid move confirm hua.
        Black STOPS → White STARTS.
        main.py mein human move confirm ke baad call karo.
        """
        with self._lock:
            if self.status == "timeout":
                return
            self._flush()
            self.black_moves += 1
            self.active       = "white"
            self.status       = "running"
            self._tick_start  = time.time()
        self._push_redis()
        self._save_to_file()
        logger.info(
            "[Timer] ⚫→⚪  Black move #%d | White: %s | Black: %s",
            self.black_moves, _fmt(self.white_ms), _fmt(self.black_ms)
        )

    def pause(self):
        with self._lock:
            if self.status == "running":
                self._flush()
                self.status = "paused"
        self._push_redis()
        self._save_to_file()
        logger.info("[Timer] ⏸  Paused")

    def resume(self):
        with self._lock:
            if self.status == "paused" and self.active != "stopped":
                self.status      = "running"
                self._tick_start = time.time()
        self._push_redis()
        self._save_to_file()
        logger.info("[Timer] ▶️  Resumed — %s clock running", self.active)

    def get_times(self):
        """
        Live (white_ms, black_ms) return karo.
        Ongoing tick bhi include hoti hai — hamesha accurate.
        """
        with self._lock:
            white = self.white_ms
            black = self.black_ms
            if self._tick_start and self.status == "running":
                elapsed = int((time.time() - self._tick_start) * 1000)
                if self.active == "white":
                    white = max(0, white - elapsed)
                elif self.active == "black":
                    black = max(0, black - elapsed)
            return white, black

    def stop(self):
        """Background thread band karo. Game end summary print karo."""
        self._stop_event.set()

        white_ms, black_ms = self.get_times()
        print("\n" + "=" * 50)
        print("  ♟️  CHESS GAME ENDED")
        print(f"  ⚪ White (Robot) time left : {_fmt(white_ms)}")
        print(f"  ⚫ Black (Human) time left : {_fmt(black_ms)}")
        if self.winner:
            print(f"  🏆 Winner (by time) : {self.winner.upper()}")
        print(f"  ⚪ White total moves : {self.white_moves}")
        print(f"  ⚫ Black total moves : {self.black_moves}")
        print("=" * 50 + "\n")
        logger.info("[Timer] 🛑 Stopped")

    # ── INTERNAL ──────────────────────────────────────────

    def _flush(self):
        """Active player ka elapsed time deduct karo. Lock ke andar call karo."""
        if self._tick_start is None or self.status != "running":
            return
        elapsed_ms       = int((time.time() - self._tick_start) * 1000)
        self._tick_start = time.time()
        if self.active == "white":
            self.white_ms = max(0, self.white_ms - elapsed_ms)
        elif self.active == "black":
            self.black_ms = max(0, self.black_ms - elapsed_ms)

    def _tick_loop(self):
        """Background thread — har second chalta hai."""
        while not self._stop_event.is_set():
            time.sleep(1)
            with self._lock:
                if self.status != "running":
                    continue
                self._flush()

                # Timeout check
                if self.active == "white" and self.white_ms <= 0:
                    self.white_ms = 0
                    self.status   = "timeout"
                    self.winner   = "black"
                    logger.warning("[Timer] ⏰ WHITE timed out — Black wins!")

                elif self.active == "black" and self.black_ms <= 0:
                    self.black_ms = 0
                    self.status   = "timeout"
                    self.winner   = "white"
                    logger.warning("[Timer] ⏰ BLACK timed out — White wins!")

            self._push_redis()
            self._save_to_file()

    def _save_to_file(self):
        """
        timer_log.txt mein har second save karo.
        GUI friend is file ko read karta hai.
        Format:
            white_fmt=09:45
            black_fmt=09:52
            active=black
            status=running
            winner=
            white_moves=3
            black_moves=2
        """
        try:
            white_ms, black_ms = self.get_times()
            content = (
                f"white_fmt={_fmt(white_ms)}\n"
                f"black_fmt={_fmt(black_ms)}\n"
                f"active={self.active}\n"
                f"status={self.status}\n"
                f"winner={self.winner}\n"
                f"white_moves={self.white_moves}\n"
                f"black_moves={self.black_moves}\n"
            )
            with open(TIMER_LOG_FILE, "w") as f:
                f.write(content)
        except Exception as e:
            logger.warning("[Timer] File save failed: %s", e)

    def _push_redis(self):
        """Sari timer state Redis mein write karo."""
        if self.redis is None:
            return
        try:
            white_ms, black_ms = self.get_times()
            pipe = self.redis.pipeline()
            pipe.set("timer:white_ms",    white_ms)
            pipe.set("timer:black_ms",    black_ms)
            pipe.set("timer:white_fmt",   _fmt(white_ms))
            pipe.set("timer:black_fmt",   _fmt(black_ms))
            pipe.set("timer:active",      self.active)
            pipe.set("timer:status",      self.status)
            pipe.set("timer:winner",      self.winner)
            pipe.set("timer:white_moves", self.white_moves)
            pipe.set("timer:black_moves", self.black_moves)
            pipe.execute()
        except Exception as e:
            logger.warning("[Timer] Redis push failed: %s", e)

    @staticmethod
    def _connect_redis():
        try:
            r = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                socket_connect_timeout=5, decode_responses=True
            )
            r.ping()
            logger.info("[Timer] ✅ Redis connected")
            return r
        except Exception as e:
            logger.error("[Timer] ❌ Redis failed: %s", e)
            return None


# ── UTILITY ───────────────────────────────────────────────

def _fmt(ms: int) -> str:
    """Milliseconds → MM:SS string."""
    total_sec = max(0, ms) // 1000
    return f"{total_sec // 60:02d}:{total_sec % 60:02d}"