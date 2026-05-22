"""feedback.py
Simple text-to-speech announcer for robotic-arm chess moves.

Usage:
    pip install pyttsx3
    python feedback.py e2e4        # announces "Moving from e2 to e4."
    python feedback.py d2 d4       # announces "Moving from d2 to d4."

The module exposes `announce_move(from_sq, to_sq)` and `announce_best_move(uci_move)`.
It prefers `pyttsx3` (offline). If unavailable, it falls back to Windows SAPI (pywin32),
and finally to printing the message.
"""
from __future__ import annotations

import threading
from typing import Optional

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except Exception:
    _HAS_PYTTSX3 = False

_engine: Optional[object] = None
_lock = threading.Lock()

def _get_engine():
    global _engine
    if _engine is None and _HAS_PYTTSX3:
        try:
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 150)
            _engine.setProperty("volume", 1.0)
        except Exception:
            _engine = None
    return _engine

def _speak_text(text: str, blocking: bool = True) -> None:
    """Speak text using available TTS engine or print as fallback."""
    if _HAS_PYTTSX3:
        eng = _get_engine()
        if eng is None:
            print(text)
            return

        def _run():
            with _lock:
                eng.say(text)
                eng.runAndWait()

        if blocking:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()
        return

    # Try Windows SAPI via pywin32 as a fallback
    try:
        import win32com.client  # type: ignore
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
        return
    except Exception:
        pass

    # Last resort: print the message so it can be read or logged
    print(text)

def announce_move(from_sq: str, to_sq: str, blocking: bool = True) -> None:
    """Announce a move like: 'Moving from d2 to d4.'

    Args:
        from_sq: source square (e.g., 'd2')
        to_sq: destination square (e.g., 'd4')
        blocking: if False, return immediately while speech runs in background
    """
    text = f"Moving from {from_sq} to {to_sq}."
    _speak_text(text, blocking=blocking)

def announce_best_move(uci_move: str, blocking: bool = True) -> None:
    """Announce a best move provided in UCI format (e.g., 'e2e4')."""
    if len(uci_move) >= 4:
        announce_move(uci_move[:2], uci_move[2:4], blocking=blocking)
    else:
        _speak_text(f"Best move is {uci_move}.", blocking=blocking)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Announce chess moves via TTS")
    parser.add_argument("squares", nargs="+", help="Either two squares (from to) or one UCI move (e2e4)")
    parser.add_argument("--non-blocking", dest="blocking", action="store_false", help="Speak non-blocking")
    args = parser.parse_args()

    if len(args.squares) == 2:
        announce_move(args.squares[0], args.squares[1], blocking=args.blocking)
    elif len(args.squares) == 1:
        announce_best_move(args.squares[0], blocking=args.blocking)
    else:
        parser.print_help()
