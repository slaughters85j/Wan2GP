"""Helper to promote a historical entry into the running top Gallery.

Used by the history browser so row clicks reuse wgp.py's existing Gallery
handlers (Extract Settings, To Video Source, To Control Video) without
duplicating their logic: we simply ensure the chosen file is present in the
live in-memory ``gen["file_list"]`` and point the selection at it, then fire
the existing handler chain.
"""

from __future__ import annotations

from typing import Any


def push_to_top_gallery(gen: dict[str, Any], path: str, settings: Any) -> int:
    """Ensure ``path`` is present in ``gen["file_list"]`` and select it.

    If the path is already present, its existing settings are kept untouched
    (we never overwrite in-memory settings the user might be working with).
    Otherwise the entry is appended with the provided settings.

    Returns the selected index within ``gen["file_list"]``.
    """
    if not isinstance(gen, dict):
        raise ValueError("gen must be a dict")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    file_list = gen.get("file_list")
    if not isinstance(file_list, list):
        file_list = []
        gen["file_list"] = file_list
    file_settings_list = gen.get("file_settings_list")
    if not isinstance(file_settings_list, list):
        file_settings_list = []
        gen["file_settings_list"] = file_settings_list

    try:
        idx = file_list.index(path)
    except ValueError:
        file_list.append(path)
        file_settings_list.append(settings if isinstance(settings, dict) else None)
        idx = len(file_list) - 1

    # Pad settings list if it's out of sync for any reason.
    while len(file_settings_list) < len(file_list):
        file_settings_list.append(None)

    gen["selected"] = idx
    # User made an explicit selection; don't auto-snap to "latest" on refresh.
    gen["last_selected"] = False
    gen["selected_video_time"] = 0.0
    return idx
