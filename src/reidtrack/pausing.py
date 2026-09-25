"""Pausing long training runs.

A run stops at the next step when Ctrl+C is pressed or a file named PAUSE appears
in its run folder. It saves its full state and exits with ``PAUSED``; running the
same command with --resume continues from where it stopped.
"""

from __future__ import annotations

import signal
from pathlib import Path

PAUSED = 3  # exit code of a paused run, so scripts can tell it from success


class PauseRequest:
    def __init__(self, run_dir: str | Path) -> None:
        self.file = Path(run_dir) / "PAUSE"
        self.file.unlink(missing_ok=True)  # a leftover request from the previous pause
        self._interrupted = False
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_interrupt)

    def _on_interrupt(self, signum, frame) -> None:
        if self._interrupted:  # a second Ctrl+C stops at once
            raise KeyboardInterrupt
        self._interrupted = True
        print("pausing after this step; press Ctrl+C again to stop at once", flush=True)

    def __bool__(self) -> bool:
        return self._interrupted or self.file.exists()

    def close(self) -> None:
        signal.signal(signal.SIGINT, self._previous)
        self.file.unlink(missing_ok=True)
