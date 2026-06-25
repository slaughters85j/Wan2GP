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

# Project root is derived from this file's location:
#   shared/utils/history_persistence.py  →  parents up 2 dirs = project root.
# This keeps the JSON on the local filesystem regardless of where the user's
# `save_path` points (network drives are unreliable; the local FS is not).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_HISTORY_PATH = os.path.join(_PROJECT_ROOT, _HISTORY_FILENAME)

_lock = threading.Lock()
_history_path: str | None = None
_save_dir: str | None = None
_loaded: bool = False
_video_entries: list[dict[str, Any]] = []
_audio_entries: list[dict[str, Any]] = []


def _log(msg: str) -> None:
    try:
        print(f"[history_persistence] {msg}")
    except Exception:
        pass


def _norm(path: Any) -> str:
    """Canonical form for path equality. Stable across slash flavor and case
    (on Windows). Returns "" for non-string inputs.
    """
    if not isinstance(path, str) or not path:
        return ""
    try:
        p = os.path.normpath(path)
    except Exception:
        p = path
    if os.name == "nt":
        p = p.lower()
    return p


def init_history_store(save_dir: str | None) -> None:
    """Locate the on-disk history JSON (always at project root) and load
    existing entries. ``save_dir`` is recorded for the inventory report only;
    the file itself lives on the local filesystem so it stays available even
    when ``save_dir`` is a flaky network share.

    Safe to call multiple times; subsequent calls are no-ops unless the
    history path itself changes (which it currently never does — the path is
    derived from this module's own location).
    """
    global _history_path, _save_dir, _loaded, _video_entries, _audio_entries
    with _lock:
        try:
            _save_dir = save_dir  # purely informational (used by inventory report)
            new_path = _DEFAULT_HISTORY_PATH
            if _loaded and new_path == _history_path:
                return
            try:
                os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
            except Exception as exc:
                _log(f"Could not ensure dir for {new_path!r}: {exc}")
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


def get_save_dir() -> str | None:
    return _save_dir


def reload_from_disk() -> tuple[int, int]:
    """Force a re-read of the JSON from disk into module memory.

    Returns ``(video_count, audio_count)`` after the reload. Used by the
    Refresh button to ensure the UI sees the latest on-disk state even if
    something outside this process touched the file.
    """
    global _video_entries, _audio_entries
    with _lock:
        if _history_path is None:
            return (len(_video_entries), len(_audio_entries))
        try:
            v, a = _read_from_disk(_history_path)
            _video_entries = v
            _audio_entries = a
        except Exception as exc:
            _log(f"reload_from_disk failed: {exc}")
        return (len(_video_entries), len(_audio_entries))


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
    target_norm = _norm(path)
    with _lock:
        try:
            entries = _entries_for(audio)
            safe_settings = settings if isinstance(settings, dict) else None
            for e in entries:
                if _norm(e.get("path")) == target_norm:
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
    target_norm = _norm(path)
    with _lock:
        try:
            entries = _entries_for(audio)
            before = len(entries)
            entries[:] = [e for e in entries if _norm(e.get("path")) != target_norm]
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
    target_norm = _norm(path)
    removed = False
    with _lock:
        try:
            entries = _entries_for(audio)
            before = len(entries)
            entries[:] = [e for e in entries if _norm(e.get("path")) != target_norm]
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
    target_norm = _norm(path)
    with _lock:
        try:
            entries = _entries_for(audio)
            for e in entries:
                if _norm(e.get("path")) == target_norm:
                    new_val = not bool(e.get("favorite", False))
                    e["favorite"] = new_val
                    _write_to_disk_locked()
                    _log(f"toggle_favorite: {path!r} -> {new_val}")
                    return new_val
            _log(f"toggle_favorite: no matching entry for {path!r} (norm={target_norm!r}); kept {len(entries)} entries unchanged")
        except Exception as exc:
            _log(f"toggle_favorite failed for {path!r}: {exc}")
    return False


def set_favorite(path: str, value: bool, *, audio: bool = False) -> None:
    if not isinstance(path, str) or not path:
        return
    target_norm = _norm(path)
    with _lock:
        try:
            entries = _entries_for(audio)
            for e in entries:
                if _norm(e.get("path")) == target_norm:
                    e["favorite"] = bool(value)
                    _write_to_disk_locked()
                    return
        except Exception as exc:
            _log(f"set_favorite failed for {path!r}: {exc}")


_MEDIA_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def disk_inventory_report() -> str:
    """Build a human-readable report of the persistence state for the
    Refresh-button debug log. Compares JSON entries to media files actually
    sitting in ``save_dir``.
    """
    lines: list[str] = []
    lines.append("=== history_persistence inventory ===")
    lines.append(f"history JSON: {_history_path!r}")
    lines.append(f"save_dir:     {_save_dir!r}")
    try:
        json_size = os.path.getsize(_history_path) if _history_path and os.path.isfile(_history_path) else -1
    except Exception:
        json_size = -1
    lines.append(f"JSON size:    {json_size} bytes")
    with _lock:
        v_count = len(_video_entries)
        a_count = len(_audio_entries)
        v_paths_norm = {_norm(e.get("path")): e.get("path") for e in _video_entries}
        a_paths_norm = {_norm(e.get("path")): e.get("path") for e in _audio_entries}
        v_fav = sum(1 for e in _video_entries if e.get("favorite"))
        a_fav = sum(1 for e in _audio_entries if e.get("favorite"))
    lines.append(f"in-memory:    {v_count} video, {a_count} audio  (favorites: {v_fav} video, {a_fav} audio)")

    missing: list[str] = []
    for orig in list(v_paths_norm.values()) + list(a_paths_norm.values()):
        if isinstance(orig, str):
            try:
                if not os.path.isfile(orig):
                    missing.append(orig)
            except Exception:
                missing.append(orig)
    lines.append(f"JSON entries with file missing on disk: {len(missing)}")
    for p in missing[:10]:
        lines.append(f"  · missing: {p}")
    if len(missing) > 10:
        lines.append(f"  · …and {len(missing) - 10} more")

    if _save_dir and os.path.isdir(_save_dir):
        try:
            on_disk: list[str] = []
            for name in os.listdir(_save_dir):
                full = os.path.join(_save_dir, name)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in _MEDIA_EXTS:
                    on_disk.append(full)
            lines.append(f"media files in save_dir: {len(on_disk)}")
            untracked = [p for p in on_disk if _norm(p) not in v_paths_norm and _norm(p) not in a_paths_norm]
            lines.append(f"on-disk files NOT in history: {len(untracked)}")
            for p in untracked[:10]:
                lines.append(f"  · untracked: {os.path.basename(p)}")
            if len(untracked) > 10:
                lines.append(f"  · …and {len(untracked) - 10} more")
        except Exception as exc:
            lines.append(f"save_dir scan failed: {exc}")
    else:
        lines.append("save_dir is not a directory; skipping disk scan.")

    lines.append("=== end inventory ===")
    return "\n".join(lines)


def sync_from_lists(file_list: list[str] | None, file_settings_list: list[Any] | None, *, audio: bool = False) -> None:
    """Replace the persisted list for the given kind with the provided lists.

    Used by bulk operations (import, clear_deleted) where a per-item diff
    would be more fragile than a full rewrite. Preserves the ``favorite`` flag
    for paths that survive the sync.
    """
    with _lock:
        try:
            target = _entries_for(audio)
            prev_fav: dict[str, bool] = {}
            for e in target:
                key = _norm(e.get("path"))
                if key:
                    prev_fav[key] = bool(e.get("favorite", False))
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
                    "favorite": prev_fav.get(_norm(path), False),
                })
            target[:] = new_entries
            _write_to_disk_locked()
        except Exception as exc:
            _log(f"sync_from_lists failed (audio={audio}): {exc}")
