"""Persistent history of generated media and their creation metadata.

Keeps a JSON sidecar so the Gradio gallery (Media Info / Late Post Processing /
Import Media) survives app restarts. Nothing here raises: failures are logged
and the app continues with an empty or partial history.

The JSON is stored in the app's save_path (e.g. ``outputs/``) which is
gitignored, so it persists across `git pull` from the upstream public repo.

Schema:
    schema_version: 2
    video_entries / audio_entries: list of
        {
          "path": str,
          "settings": dict | None,
          "favorite": bool,
        }
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any

_HISTORY_FILENAME = "_wangp_history.json"
_SCHEMA_VERSION = 2

_lock = threading.Lock()
_history_path: str | None = None
_loaded: bool = False
_video_entries: list[dict[str, Any]] = []
_audio_entries: list[dict[str, Any]] = []


def _log(msg: str) -> None:
    try:
        print(f"[history_persistence] {msg}")
    except Exception:
        pass


def init_history_store(save_dir: str | None) -> None:
    """Set the directory where the JSON lives and load existing entries.

    Safe to call multiple times; subsequent calls re-point the store only if
    the directory differs and reload from the new location.
    """
    global _history_path, _loaded, _video_entries, _audio_entries
    with _lock:
        try:
            if save_dir is None:
                save_dir = os.path.join(os.getcwd(), "outputs")
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as exc:
                _log(f"Could not create save_dir {save_dir!r}: {exc}")
            new_path = os.path.join(save_dir, _HISTORY_FILENAME)
            if _loaded and new_path == _history_path:
                return
            _history_path = new_path
            _video_entries, _audio_entries = _read_from_disk(new_path)
            _loaded = True
        except Exception as exc:
            _log(f"init_history_store failed: {exc}")
            _history_path = None
            _video_entries = []
            _audio_entries = []
            _loaded = True


def get_history_path() -> str | None:
    return _history_path


def _normalize_entry(e: Any) -> dict[str, Any] | None:
    if not isinstance(e, dict):
        return None
    path = e.get("path")
    if not isinstance(path, str) or not path:
        return None
    cfg = e.get("settings")
    fav = bool(e.get("favorite", False))
    return {
        "path": path,
        "settings": cfg if isinstance(cfg, dict) else None,
        "favorite": fav,
    }


def _read_from_disk(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path or not os.path.isfile(path):
        return [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        _log(f"Failed to read {path!r}: {exc}. Starting empty.")
        return [], []
    if not isinstance(data, dict):
        _log(f"{path!r} has unexpected structure; starting empty.")
        return [], []
    raw_videos = data.get("video_entries") or []
    raw_audios = data.get("audio_entries") or []
    videos = [v for v in (_normalize_entry(e) for e in raw_videos) if v is not None]
    audios = [v for v in (_normalize_entry(e) for e in raw_audios) if v is not None]
    return videos, audios


def _write_to_disk_locked() -> None:
    """Atomic write. Caller must hold ``_lock``."""
    if _history_path is None:
        return
    data = {
        "schema_version": _SCHEMA_VERSION,
        "video_entries": _video_entries,
        "audio_entries": _audio_entries,
    }
    try:
        target_dir = os.path.dirname(_history_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="._wangp_history_", suffix=".json", dir=target_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(data, tmp, indent=2, default=str)
            os.replace(tmp_path, _history_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise
    except Exception as exc:
        _log(f"Failed to write {_history_path!r}: {exc}")


def _entries_for(audio: bool) -> list[dict[str, Any]]:
    return _audio_entries if audio else _video_entries


def load_entries(audio: bool = False) -> tuple[list[str], list[dict[str, Any] | None]]:
    """Return (paths, settings_list) filtered to files that still exist on disk.

    Used to seed the top Gallery. The history browser uses
    ``snapshot_entries()`` instead so it can still display entries whose
    underlying file is missing.
    """
    with _lock:
        entries = _entries_for(audio)
        paths: list[str] = []
        settings: list[dict[str, Any] | None] = []
        for e in entries:
            path = e.get("path")
            if not isinstance(path, str):
                continue
            try:
                exists = os.path.isfile(path)
            except Exception:
                exists = False
            if not exists:
                continue
            cfg = e.get("settings")
            paths.append(path)
            settings.append(cfg if isinstance(cfg, dict) else None)
        return paths, settings


def snapshot_entries(audio: bool = False) -> list[dict[str, Any]]:
    """Return a shallow copy of all entries (including missing-file ones).

    Each entry: {path, settings, favorite, exists}. Caller must not mutate.
    """
    with _lock:
        out: list[dict[str, Any]] = []
        for e in _entries_for(audio):
            path = e.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                exists = os.path.isfile(path)
            except Exception:
                exists = False
            out.append({
                "path": path,
                "settings": e.get("settings") if isinstance(e.get("settings"), dict) else None,
                "favorite": bool(e.get("favorite", False)),
                "exists": exists,
            })
        return out


def append_entry(path: str, settings: Any, *, audio: bool = False) -> None:
    if not isinstance(path, str) or not path:
        return
    with _lock:
        try:
            entries = _entries_for(audio)
            safe_settings = settings if isinstance(settings, dict) else None
            for e in entries:
                if e.get("path") == path:
                    if safe_settings is not None:
                        e["settings"] = safe_settings
                    _write_to_disk_locked()
                    return
            entries.append({"path": path, "settings": safe_settings, "favorite": False})
            _write_to_disk_locked()
        except Exception as exc:
            _log(f"append_entry failed for {path!r}: {exc}")


def replace_last_entry(new_path: str, new_settings: Any, *, audio: bool = False) -> None:
    """Used when a sliding-window output replaces the previous last file."""
    if not isinstance(new_path, str) or not new_path:
        return
    with _lock:
        try:
            entries = _entries_for(audio)
            safe_settings = new_settings if isinstance(new_settings, dict) else None
            if entries:
                # Preserve favorite state across sliding-window replace.
                prev_fav = bool(entries[-1].get("favorite", False))
                entries[-1] = {"path": new_path, "settings": safe_settings, "favorite": prev_fav}
            else:
                entries.append({"path": new_path, "settings": safe_settings, "favorite": False})
            _write_to_disk_locked()
        except Exception as exc:
            _log(f"replace_last_entry failed for {new_path!r}: {exc}")


def remove_entry(path: str, *, audio: bool = False) -> None:
    """Remove the JSON entry only (does not touch the file on disk)."""
    if not isinstance(path, str) or not path:
        return
    with _lock:
        try:
            entries = _entries_for(audio)
            before = len(entries)
            entries[:] = [e for e in entries if e.get("path") != path]
            if len(entries) != before:
                _write_to_disk_locked()
        except Exception as exc:
            _log(f"remove_entry failed for {path!r}: {exc}")


def delete_entry(path: str, *, audio: bool = False, delete_file: bool = True, thumb_cb=None) -> bool:
    """Delete the JSON entry plus the underlying file (if present) and its thumbnail.

    ``thumb_cb`` is an optional ``callable(path) -> None`` that the caller
    can pass to also wipe cached thumbnails. Returns True if the entry was
    found and removed from the JSON.
    """
    if not isinstance(path, str) or not path:
        return False
    removed = False
    with _lock:
        try:
            entries = _entries_for(audio)
            before = len(entries)
            entries[:] = [e for e in entries if e.get("path") != path]
            removed = len(entries) != before
            if removed:
                _write_to_disk_locked()
        except Exception as exc:
            _log(f"delete_entry (json remove) failed for {path!r}: {exc}")
    if delete_file:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as exc:
            _log(f"delete_entry (file remove) failed for {path!r}: {exc}")
    if thumb_cb is not None:
        try:
            thumb_cb(path)
        except Exception as exc:
            _log(f"delete_entry (thumb cleanup) failed for {path!r}: {exc}")
    return removed


def toggle_favorite(path: str, *, audio: bool = False) -> bool:
    """Flip the favorite flag for ``path``. Returns the new value (or False if
    the entry was not found)."""
    if not isinstance(path, str) or not path:
        return False
    with _lock:
        try:
            entries = _entries_for(audio)
            for e in entries:
                if e.get("path") == path:
                    new_val = not bool(e.get("favorite", False))
                    e["favorite"] = new_val
                    _write_to_disk_locked()
                    return new_val
        except Exception as exc:
            _log(f"toggle_favorite failed for {path!r}: {exc}")
    return False


def set_favorite(path: str, value: bool, *, audio: bool = False) -> None:
    if not isinstance(path, str) or not path:
        return
    with _lock:
        try:
            entries = _entries_for(audio)
            for e in entries:
                if e.get("path") == path:
                    e["favorite"] = bool(value)
                    _write_to_disk_locked()
                    return
        except Exception as exc:
            _log(f"set_favorite failed for {path!r}: {exc}")


def sync_from_lists(file_list: list[str] | None, file_settings_list: list[Any] | None, *, audio: bool = False) -> None:
    """Replace the persisted list for the given kind with the provided lists.

    Used by bulk operations (import, clear_deleted) where a per-item diff
    would be more fragile than a full rewrite. Preserves the ``favorite`` flag
    for paths that survive the sync.
    """
    with _lock:
        try:
            target = _entries_for(audio)
            prev_fav = {e.get("path"): bool(e.get("favorite", False)) for e in target if isinstance(e.get("path"), str)}
            new_entries: list[dict[str, Any]] = []
            paths = file_list or []
            settings = file_settings_list or []
            for i, path in enumerate(paths):
                if not isinstance(path, str) or not path:
                    continue
                cfg = settings[i] if i < len(settings) else None
                new_entries.append({
                    "path": path,
                    "settings": cfg if isinstance(cfg, dict) else None,
                    "favorite": prev_fav.get(path, False),
                })
            target[:] = new_entries
            _write_to_disk_locked()
        except Exception as exc:
            _log(f"sync_from_lists failed (audio={audio}): {exc}")
