from __future__ import annotations

import threading
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ProfileObserver:
    """Watch profile data and model roots; reload only the affected profile."""

    def __init__(self, roots: dict[str, list[Path]], callback):
        self.observer = Observer()
        self.roots = roots
        self.callback = callback
        self._lock = threading.Lock()

    def start(self) -> None:
        for profile, paths in self.roots.items():
            for path in paths:
                if path.exists():
                    handler = _Handler(profile, self.callback, self._lock)
                    self.observer.schedule(handler, str(path), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()


class _Handler(FileSystemEventHandler):
    def __init__(self, profile, callback, lock):
        self.profile, self.callback, self.lock = profile, callback, lock

    def on_any_event(self, event):
        if event.is_directory:
            return
        with self.lock:
            self.callback(self.profile, Path(event.src_path), event.event_type)
