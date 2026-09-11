"""Shutdown helper for Gradio's served-file cache.

Gradio copies every file it serves to the browser into its cache folder
(``save_file_to_cache`` does a ``shutil.copy2`` into ``<cache>/<hash>/<name>``)
and never removes them, so a session that displays a few dozen generated clips
can leave several GB behind. WanGP builds its Blocks without ``delete_cache``,
which means nothing clears those copies, not even on exit.

``prompt_purge_on_exit`` is called once after the Gradio server stops and offers
to delete them. It is deliberately opt-in per shutdown: deleting a cached copy
while its clip is still on screen in a browser tab breaks playback of that item
until the page re-renders.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path


def gradio_cache_dir() -> Path:
    """Cache folder Gradio serves from (mirrors gradio.utils.get_upload_folder)."""
    override = str(os.environ.get("GRADIO_TEMP_DIR", "")).strip()
    if len(override) > 0:
        return Path(override).resolve()
    return (Path(tempfile.gettempdir()) / "gradio").resolve()


def measure_cache(root: Path) -> tuple[int, int]:
    """Return (file count, total bytes) under root, ignoring unreadable entries."""
    files = 0
    total = 0
    for directory, _subdirs, names in os.walk(root, onerror=lambda _error: None):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(directory, name))
                files += 1
            except OSError:
                pass
    return files, total


def _entry_size(entry: Path) -> int:
    try:
        if entry.is_dir():
            return measure_cache(entry)[1]
        return entry.stat().st_size
    except OSError:
        return 0


def purge_cache(root: Path) -> tuple[int, int]:
    """Delete the cache's contents, keeping the folder. Returns (bytes freed, entries skipped)."""
    freed = 0
    skipped = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0, 0
    for entry in entries:
        size = _entry_size(entry)
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            freed += size
        except OSError:
            # Still open by the server or a browser tab; leave it for next time.
            skipped += 1
    return freed, skipped


def _is_safe_target(root: Path) -> bool:
    """Only ever touch Gradio's own cache, or an explicitly configured override."""
    if len(str(os.environ.get("GRADIO_TEMP_DIR", "")).strip()) > 0:
        return True
    return root.name.lower() == "gradio"


def prompt_purge_on_exit() -> None:
    """Offer to clear Gradio's file cache. Safe to call when nothing is cached."""
    try:
        root = gradio_cache_dir()
        if not _is_safe_target(root) or not root.is_dir():
            return
        files, total = measure_cache(root)
        if files == 0:
            return
        if not sys.stdin or not sys.stdin.isatty():
            print(f"\nGradio cache holds {files} files ({total / 1e6:.1f} MB) in {root}")
            print("Not an interactive terminal, leaving it in place.")
            return
        print(f"\nGradio cache: {files} files, {total / 1e6:.1f} MB")
        print(f"  {root}")
        try:
            answer = input("Purge it now? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("Left in place.")
            return
        if answer not in {"y", "yes"}:
            print("Left in place.")
            return
        freed, skipped = purge_cache(root)
        detail = f" ({skipped} still in use, skipped)" if skipped > 0 else ""
        print(f"Freed {freed / 1e6:.1f} MB{detail}.")
    except Exception as exc:
        # Never let cleanup noise mask a real shutdown.
        print(f"[gradio_cache] purge skipped: {exc}")


if __name__ == "__main__":
    prompt_purge_on_exit()
