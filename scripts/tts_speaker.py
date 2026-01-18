from __future__ import annotations

import threading
import queue
from typing import Optional

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


class TTSSpeaker:
    def __init__(self, enabled: bool = True, rate: int = 190, volume: float = 1.0):
        self.enabled = bool(enabled) and (pyttsx3 is not None)
        self._q: "queue.Queue[str]" = queue.Queue()
        self._th: Optional[threading.Thread] = None
        self._rate = int(rate)
        self._volume = float(volume)

        if self.enabled:
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()

    def say(self, text: str):
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return
        while True:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(text)

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while True:
            text = self._q.get()
            try:
                engine.stop()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
