from __future__ import annotations
import threading
import queue
import time

class ChineseSpeaker:
    """Non-blocking Mandarin TTS wrapper.

    pyttsx3 uses the operating system's installed speech engine. If a Chinese
    voice is unavailable, the app keeps running silently instead of crashing.
    """

    def __init__(self, enabled: bool = True, cooldown: float = 1.2):
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_spoken = 0.0
        self.q = queue.Queue()
        self.thread = None

        if enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def speak(self, text: str) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if now - self.last_spoken < self.cooldown:
            return False
        self.last_spoken = now
        self.q.put(text)
        return True

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            # Prefer voices whose metadata contains Mandarin/Chinese markers.
            preferred = []
            for v in voices:
                blob = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')} {getattr(v, 'languages', '')}".lower()
                if any(k in blob for k in ("zh", "chinese", "mandarin", "huihui", "hanhan")):
                    preferred.append(v)
            if preferred:
                engine.setProperty("voice", preferred[0].id)
            while True:
                text = self.q.get()
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
        except Exception:
            # Audio is optional; camera recognition must still work.
            return

    def close(self):
        if self.enabled:
            self.q.put(None)
