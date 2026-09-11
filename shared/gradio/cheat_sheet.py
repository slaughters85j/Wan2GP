"""Header button that opens the local LoRA cheat sheet in a detached window.

The sheet is a plain HTML file kept at the project root (gitignored, user-owned).
Gradio serves it inline through its ``/gradio_api/file=`` route as long as the
file is in ``allowed_paths`` - see ``allowed_paths()`` below. The button is a
``gr.HTML`` (not ``gr.Markdown``) so the ``onclick`` popup handler survives
Gradio's sanitizer.
"""

import os
from html import escape

CHEAT_SHEET_FILE = "LORA_CHEAT_SHEET.html"
_WINDOW_NAME = "wangp_lora_cheat_sheet"
_WINDOW_FEATURES = "popup,width=1100,height=900,resizable,scrollbars"


def _sheet_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, CHEAT_SHEET_FILE)


def exists() -> bool:
    return os.path.isfile(_sheet_path())


def allowed_paths() -> list[str]:
    """Extra entries for ``demo.launch(allowed_paths=...)``; empty when the sheet is absent."""
    return [CHEAT_SHEET_FILE] if exists() else []


def get_css() -> str:
    # The gr.HTML block sits right under the centered title; collapse the block to
    # zero height and pull the button up onto the title line, flush right.
    return """
.wangp-cheat-sheet-block { height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important;
  border: none !important; background: none !important; overflow: visible !important; }
.wangp-cheat-sheet-block > * { overflow: visible !important; }
.wangp-cheat-sheet { display: flex; justify-content: flex-end; padding-right: 8px; transform: translateY(-2.9em);
  position: relative; z-index: 20; pointer-events: none; }
.wangp-cheat-sheet a { pointer-events: auto; text-decoration: none; }
.wangp-cheat-sheet button { font-size: 0.85em; padding: 4px 10px; border-radius: 8px; cursor: pointer; white-space: nowrap;
  border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); color: var(--body-text-color); }
.wangp-cheat-sheet button:hover { background: var(--background-fill-primary); border-color: var(--color-accent); }
"""


def get_html() -> str:
    """Button markup, or an empty string when there is no sheet to open."""
    if not exists():
        return ""
    url = f"/gradio_api/file={CHEAT_SHEET_FILE}"
    onclick = escape(f"window.open('{url}', '{_WINDOW_NAME}', '{_WINDOW_FEATURES}'); return false;", quote=True)
    return (
        f'<div class="wangp-cheat-sheet"><a href="{url}" target="_blank" onclick="{onclick}">'
        f'<button type="button" title="Open {escape(CHEAT_SHEET_FILE)} in a separate window">'
        f'&#128218; LoRA Cheat Sheet</button></a></div>'
    )
