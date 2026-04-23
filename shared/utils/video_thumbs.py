"""Cached thumbnail extraction for the history browser.

Extracts frame 0 of a video via ffmpeg, scales it down, and caches as JPEG
under ``<save_dir>/_thumbs/``. Safe on failure: logs and returns None.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Optional

_THUMB_DIRNAME = "_thumbs"
_THUMB_WIDTH = 256
_DEFAULT_TIMEOUT_S = 20


def _log(msg: str) -> None:
    try:
        print(f"[video_thumbs] {msg}")
    except Exception:
        pass


def thumbs_dir(save_dir: str) -> str:
    return os.path.join(save_dir, _THUMB_DIRNAME)


def thumb_path_for(video_path: str, save_dir: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    # Prefix with a hash so different files with the same basename don't
    # collide, and truncate the human-readable tail so Windows path limits
    # are not an issue.
    h = hashlib.md5(video_path.encode("utf-8", errors="ignore")).hexdigest()[:10]
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]
    return os.path.join(thumbs_dir(save_dir), f"{h}_{safe_name}.jpg")


def get_or_make_thumbnail(video_path: str, save_dir: str, width: int = _THUMB_WIDTH) -> Optional[str]:
    """Return a path to a cached thumbnail, generating it if missing.

    Returns None if the source file is absent, the extraction fails, or
    ffmpeg is unavailable.
    """
    try:
        if not isinstance(video_path, str) or not video_path:
            return None
        if not os.path.isfile(video_path):
            return None
        out_path = thumb_path_for(video_path, save_dir)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", "0",
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "4",
            out_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_DEFAULT_TIMEOUT_S,
        )
        if result.returncode != 0:
            _log(f"ffmpeg exited {result.returncode} for {video_path!r}: {result.stderr.decode(errors='ignore')[:200]}")
            return None
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return None
    except FileNotFoundError:
        _log("ffmpeg not found on PATH; thumbnails disabled")
        return None
    except subprocess.TimeoutExpired:
        _log(f"ffmpeg timed out for {video_path!r}")
        return None
    except Exception as exc:
        _log(f"thumbnail generation failed for {video_path!r}: {exc}")
        return None


def delete_thumbnail_for(video_path: str, save_dir: str) -> None:
    """Best-effort cleanup of a single cached thumbnail."""
    try:
        p = thumb_path_for(video_path, save_dir)
        if os.path.isfile(p):
            os.remove(p)
    except Exception as exc:
        _log(f"delete_thumbnail_for {video_path!r} failed: {exc}")
