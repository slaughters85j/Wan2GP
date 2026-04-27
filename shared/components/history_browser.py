"""Paginated history browser for WanGP.

Sits below the Queue Management accordion. Lets the user browse the full
persisted history (well past the ``clear_file_list`` cap of the top Gallery)
without loading every video as a heavyweight ``gr.Video`` player. Each row
shows a cached first-frame thumbnail plus the prompt and a small metadata
line, with action buttons that reuse wgp.py's existing handlers by routing
through ``push_to_top_gallery``.

The component is self-contained: ``build_ui`` constructs the Gradio widgets,
``wire_events`` binds them. wgp.py only needs to plumb references to the top
Gallery, its ``last_choice`` tracker, and three hidden "trigger" textboxes
that fold the row actions into the existing Extract Settings / To Video
Source / To Control Video chains.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gradio as gr

from shared.utils import history_persistence, video_thumbs
from shared.utils.gallery_push import push_to_top_gallery


PAGE_SIZE = 10
PROMPT_PREVIEW_CHARS = 260
STAR_ON = "★"
STAR_OFF = "☆"
DELETE_LABEL = "Delete"
DELETE_CONFIRM_LABEL = "⚠ Confirm Delete"
DELETE_CONFIRM_WINDOW_S = 3.0
LATEST_BADGE_HTML = "<span class='wgp-history-latest-badge'>LATEST</span>"

# Element-class names — kept in sync with shared/gradio/ui_styles.css
_FAV_OFF_CLASS = "wgp-history-fav"
_FAV_ON_CLASS = "wgp-history-fav-on"
_DELETE_BASE_CLASS = "wgp-history-delete"
_DELETE_ARMED_CLASS = "wgp-history-delete-armed"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _log(msg: str) -> None:
    try:
        print(f"[history_browser] {msg}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Component bundles
# ---------------------------------------------------------------------------


@dataclass
class _RowRefs:
    row: gr.Row
    thumb: gr.Image
    prompt_md: gr.Markdown
    meta_md: gr.Markdown
    fav_btn: gr.Button
    play_btn: gr.Button
    extract_btn: gr.Button
    to_source_btn: gr.Button
    to_control_btn: gr.Button
    delete_btn: gr.Button
    cancel_btn: gr.Button          # only visible when delete is armed
    path_state: gr.State          # str
    exists_state: gr.State         # bool
    delete_armed_state: gr.State   # float: armed timestamp (0.0 means not armed)


@dataclass
class HistoryBrowserUI:
    accordion: gr.Accordion
    # Filter controls
    favorites_only: gr.Checkbox
    model_filter: gr.Dropdown
    lora_filter: gr.Dropdown
    search_box: gr.Textbox
    refresh_btn: gr.Button
    # Nav
    prev_btn: gr.Button
    next_btn: gr.Button
    page_info_md: gr.Markdown
    empty_md: gr.Markdown
    # Rows
    rows: list[_RowRefs]
    # Paging state
    current_page: gr.State
    # Hidden external triggers (wgp.py wires these into existing chains)
    extract_trigger: gr.Text
    to_source_trigger: gr.Text
    to_control_trigger: gr.Text
    # Shared config
    save_dir: str
    missing_placeholder: Optional[str]


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------


def _fmt_prompt(settings: Optional[dict]) -> str:
    if not isinstance(settings, dict):
        return "_(no settings)_"
    p = settings.get("prompt")
    if not isinstance(p, str) or not p.strip():
        return "_(no prompt)_"
    p = p.strip()
    if len(p) > PROMPT_PREVIEW_CHARS:
        p = p[:PROMPT_PREVIEW_CHARS].rstrip() + "…"
    return p


def _fmt_meta(settings: Optional[dict], path: str, exists: bool, *, is_latest: bool = False) -> str:
    parts: list[str] = []
    if is_latest:
        parts.append(LATEST_BADGE_HTML)
    if not exists:
        parts.append("<span style='color:#d06060'>**file missing**</span>")
    mt = settings.get("model_type") if isinstance(settings, dict) else None
    if isinstance(mt, str) and mt:
        parts.append(f"`{mt}`")
    if isinstance(settings, dict):
        loras = settings.get("activated_loras")
        if isinstance(loras, list) and loras:
            names = []
            for l in loras:
                if isinstance(l, str):
                    b, _ = os.path.splitext(os.path.basename(l))
                    names.append(b)
            if names:
                shown = ", ".join(names[:3])
                extra = f" +{len(names)-3}" if len(names) > 3 else ""
                parts.append(f"LoRAs: {shown}{extra}")
        seed = settings.get("seed")
        if isinstance(seed, int):
            parts.append(f"seed {seed}")
        res = settings.get("resolution")
        if isinstance(res, str) and res:
            parts.append(res)
    parts.append(f"<span style='opacity:0.55;font-family:monospace;font-size:0.85em'>{os.path.basename(path)}</span>")
    return " · ".join(parts)


def build_ui(
    save_dir: str,
    missing_placeholder: Optional[str] = None,
    accordion_label: str = "Full History",
) -> HistoryBrowserUI:
    """Construct the Gradio component tree. Must be called inside a Blocks."""
    rows: list[_RowRefs] = []
    with gr.Accordion(accordion_label, open=False) as accordion:
        with gr.Row():
            favorites_only = gr.Checkbox(label="Favorites only", value=False, scale=0, min_width=140)
            model_filter = gr.Dropdown(label="Model", choices=[], value=[], multiselect=True, scale=1, interactive=True)
            lora_filter = gr.Dropdown(label="LoRAs", choices=[], value=[], multiselect=True, scale=2, interactive=True)
            search_box = gr.Textbox(label="Search prompt", value="", scale=2, placeholder="substring match…")
            refresh_btn = gr.Button("⟳ Refresh", size="sm", scale=0, min_width=100)

        empty_md = gr.Markdown("_No history entries match the current filters._", visible=False)

        for i in range(PAGE_SIZE):
            with gr.Row(visible=False, equal_height=True) as row:
                with gr.Column(scale=0, min_width=180):
                    thumb = gr.Image(
                        value=None,
                        show_label=False,
                        interactive=False,
                        height=110,
                        width=180,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        container=False,
                    )
                with gr.Column(scale=4):
                    prompt_md = gr.Markdown("", elem_classes="wgp-history-prompt")
                    meta_md = gr.Markdown("", elem_classes="wgp-history-meta")
                with gr.Column(scale=0, min_width=60):
                    fav_btn = gr.Button(STAR_OFF, size="sm", min_width=44, elem_classes=[_FAV_OFF_CLASS])
                with gr.Column(scale=0, min_width=580):
                    with gr.Row():
                        play_btn = gr.Button("Play", size="sm", min_width=70)
                        extract_btn = gr.Button("Extract Settings", size="sm", min_width=130)
                        to_source_btn = gr.Button("To Video Source", size="sm", min_width=120)
                        to_control_btn = gr.Button("To Control Video", size="sm", min_width=130)
                        delete_btn = gr.Button(DELETE_LABEL, size="sm", min_width=70, variant="secondary", elem_classes=[_DELETE_BASE_CLASS])
                        cancel_btn = gr.Button("Cancel", size="sm", min_width=70, variant="secondary", visible=False)
                path_state = gr.State("")
                exists_state = gr.State(False)
                delete_armed_state = gr.State(0.0)
            rows.append(_RowRefs(
                row=row,
                thumb=thumb,
                prompt_md=prompt_md,
                meta_md=meta_md,
                fav_btn=fav_btn,
                play_btn=play_btn,
                extract_btn=extract_btn,
                to_source_btn=to_source_btn,
                to_control_btn=to_control_btn,
                delete_btn=delete_btn,
                cancel_btn=cancel_btn,
                path_state=path_state,
                exists_state=exists_state,
                delete_armed_state=delete_armed_state,
            ))

        with gr.Row():
            prev_btn = gr.Button("← Prev", size="sm", scale=0, min_width=90)
            page_info_md = gr.Markdown("Page 1 / 1", elem_classes="wgp-history-page")
            next_btn = gr.Button("Next →", size="sm", scale=0, min_width=90)

        current_page = gr.State(0)
        extract_trigger = gr.Text(visible=False, value="")
        to_source_trigger = gr.Text(visible=False, value="")
        to_control_trigger = gr.Text(visible=False, value="")

    return HistoryBrowserUI(
        accordion=accordion,
        favorites_only=favorites_only,
        model_filter=model_filter,
        lora_filter=lora_filter,
        search_box=search_box,
        refresh_btn=refresh_btn,
        prev_btn=prev_btn,
        next_btn=next_btn,
        page_info_md=page_info_md,
        empty_md=empty_md,
        rows=rows,
        current_page=current_page,
        extract_trigger=extract_trigger,
        to_source_trigger=to_source_trigger,
        to_control_trigger=to_control_trigger,
        save_dir=save_dir,
        missing_placeholder=missing_placeholder,
    )


# ---------------------------------------------------------------------------
# Filtering / pagination
# ---------------------------------------------------------------------------


def _apply_filters(
    entries: list[dict],
    favorites_only: bool,
    model_filter: list[str] | None,
    lora_filter: list[str] | None,
    search: str | None,
) -> list[dict]:
    out = entries
    if favorites_only:
        out = [e for e in out if e.get("favorite")]
    if model_filter:
        mset = set(model_filter)
        out = [e for e in out if (e.get("settings") or {}).get("model_type") in mset]
    if lora_filter:
        lset = set(lora_filter)
        def _row_loras(e: dict) -> set[str]:
            loras = (e.get("settings") or {}).get("activated_loras") or []
            names = set()
            for l in loras:
                if isinstance(l, str):
                    names.add(l)
                    b = os.path.basename(l)
                    names.add(b)
                    names.add(os.path.splitext(b)[0])
            return names
        out = [e for e in out if _row_loras(e) & lset]
    if search:
        s = search.strip().lower()
        if s:
            def _match(e: dict) -> bool:
                p = (e.get("settings") or {}).get("prompt") or ""
                return isinstance(p, str) and s in p.lower()
            out = [e for e in out if _match(e)]
    return out


def _distinct_models(entries: list[dict]) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for e in entries:
        mt = (e.get("settings") or {}).get("model_type")
        if isinstance(mt, str) and mt and mt not in seen_set:
            seen_set.add(mt)
            seen.append(mt)
    return sorted(seen)


def _distinct_loras(entries: list[dict]) -> list[str]:
    seen_set: set[str] = set()
    for e in entries:
        loras = (e.get("settings") or {}).get("activated_loras") or []
        for l in loras:
            if isinstance(l, str):
                b, _ = os.path.splitext(os.path.basename(l))
                if b:
                    seen_set.add(b)
    return sorted(seen_set)


def _lora_label_to_match_values(labels: list[str]) -> list[str]:
    """The Dropdown stores human-readable LoRA basenames; expand to all
    forms that might appear in the stored settings (full path, basename with
    ext, basename without)."""
    return list(labels)


def _render_page(
    ui: HistoryBrowserUI,
    page: int,
    favorites_only: bool,
    model_filter: list[str] | None,
    lora_filter: list[str] | None,
    search: str | None,
) -> list[Any]:
    """Compute every component update for a given page + filter combo."""
    entries_all = history_persistence.snapshot_entries(audio=False)
    filtered = _apply_filters(entries_all, favorites_only, model_filter, lora_filter, search)
    # The most-recent entry is the LAST one ever appended (regardless of
    # filters). Tagging by path so the badge follows the same item even when
    # the user is filtering or paginating.
    latest_path = entries_all[-1]["path"] if entries_all else None
    total = len(filtered)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * PAGE_SIZE
    slice_ = filtered[start:start + PAGE_SIZE]

    updates: list[Any] = []
    for i in range(PAGE_SIZE):
        if i < len(slice_):
            e = slice_[i]
            path = e["path"]
            exists = bool(e.get("exists"))
            settings = e.get("settings") or {}
            fav = bool(e.get("favorite"))
            is_image = _is_image(path)
            is_video = _is_video(path)

            # Thumbnail
            thumb_val: Any
            if exists and is_image:
                thumb_val = path
            elif exists and is_video:
                thumb_val = video_thumbs.get_or_make_thumbnail(path, ui.save_dir) or ui.missing_placeholder
            else:
                thumb_val = ui.missing_placeholder

            row_visible = True
            is_latest = latest_path is not None and path == latest_path
            prompt_md = _fmt_prompt(settings)
            meta_md = _fmt_meta(settings, path, exists, is_latest=is_latest)
            fav_label = STAR_ON if fav else STAR_OFF
            fav_classes = [_FAV_ON_CLASS] if fav else [_FAV_OFF_CLASS]
            # Buttons: disable ops that require a live file / make sense per type
            can_play = exists
            can_extract = True  # settings can be extracted even if file missing
            can_to_video = exists and is_video
            # "To Control Video" accepts video inputs only
            can_to_control = exists and is_video
            updates.extend([
                gr.update(visible=row_visible),   # row
                gr.update(value=thumb_val),        # thumb
                gr.update(value=prompt_md),        # prompt
                gr.update(value=meta_md),          # meta
                gr.update(value=fav_label, elem_classes=fav_classes),  # fav_btn
                gr.update(interactive=can_play),   # play_btn
                gr.update(interactive=can_extract),# extract_btn
                gr.update(interactive=can_to_video),  # to_source_btn
                gr.update(interactive=can_to_control), # to_control_btn
                gr.update(value=DELETE_LABEL, variant="secondary", elem_classes=[_DELETE_BASE_CLASS]),  # delete_btn (reset)
                gr.update(visible=False),          # cancel_btn (hidden unless armed)
                path,                              # path_state
                exists,                            # exists_state
                0.0,                               # delete_armed_state reset
            ])
        else:
            updates.extend([
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=STAR_OFF, elem_classes=[_FAV_OFF_CLASS]),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=DELETE_LABEL, variant="secondary", elem_classes=[_DELETE_BASE_CLASS]),
                gr.update(visible=False),
                "",
                False,
                0.0,
            ])

    # Filter choices
    model_choices = _distinct_models(entries_all)
    lora_choices = _distinct_loras(entries_all)
    kept_model = [m for m in (model_filter or []) if m in model_choices]
    kept_lora = [l for l in (lora_filter or []) if l in lora_choices]

    updates.append(gr.update(choices=model_choices, value=kept_model))
    updates.append(gr.update(choices=lora_choices, value=kept_lora))

    # Page info and nav
    updates.append(gr.update(value=f"Page {page + 1} / {pages}  ·  {total} item" + ("" if total == 1 else "s")))
    updates.append(gr.update(interactive=page > 0))
    updates.append(gr.update(interactive=page < pages - 1))

    # Empty state
    updates.append(gr.update(visible=total == 0))

    # Current page state
    updates.append(page)

    return updates


def _render_outputs(ui: HistoryBrowserUI) -> list[Any]:
    outs: list[Any] = []
    for r in ui.rows:
        outs.extend([
            r.row, r.thumb, r.prompt_md, r.meta_md,
            r.fav_btn, r.play_btn, r.extract_btn, r.to_source_btn, r.to_control_btn, r.delete_btn, r.cancel_btn,
            r.path_state, r.exists_state, r.delete_armed_state,
        ])
    outs.extend([ui.model_filter, ui.lora_filter, ui.page_info_md, ui.prev_btn, ui.next_btn, ui.empty_md, ui.current_page])
    return outs


# ---------------------------------------------------------------------------
# Event wiring
# ---------------------------------------------------------------------------


def wire_events(
    ui: HistoryBrowserUI,
    *,
    state: gr.State,
    top_gallery: gr.Gallery,
    last_choice: gr.Number,
    output_trigger: Optional[gr.Text] = None,
    get_gen_info: Callable[[Any], dict],
    select_gallery_tab_to_video: Optional[Callable[[], Any]] = None,
):
    """Bind all row/nav/filter events.

    Args:
        state: the wgp session state.
        top_gallery: the main Gallery component ('output' in wgp.py).
        last_choice: the hidden gr.Number tracking the top Gallery's selection.
        output_trigger: if provided, its ``.change`` refreshes the browser.
        get_gen_info: wgp.py's ``get_gen_info`` function.
    """

    def _refresh_fn(page, fav_only, m_filter, l_filter, search):
        try:
            return _render_page(ui, int(page or 0), bool(fav_only), list(m_filter or []), list(l_filter or []), str(search or ""))
        except Exception as exc:
            _log(f"render failed: {exc}")
            # Best-effort: hide everything rather than raising into the UI.
            return _render_page(ui, 0, False, [], [], "")

    def _refresh_with_debug(page, fav_only, m_filter, l_filter, search):
        """Refresh-button handler. Reloads the JSON from disk so the user sees
        the freshest state, dumps an inventory report (in-memory vs disk) to
        the console, then renders the page normally."""
        try:
            mem_before_v, mem_before_a = len(history_persistence.snapshot_entries(False)), len(history_persistence.snapshot_entries(True))
            disk_v, disk_a = history_persistence.reload_from_disk()
            delta_v = disk_v - mem_before_v
            delta_a = disk_a - mem_before_a
            _log(f"refresh: reloaded JSON. video {mem_before_v}→{disk_v} (Δ{delta_v:+d}), audio {mem_before_a}→{disk_a} (Δ{delta_a:+d})")
            try:
                print(history_persistence.disk_inventory_report())
            except Exception as exc:
                _log(f"refresh: inventory report failed: {exc}")
        except Exception as exc:
            _log(f"refresh debug step failed: {exc}")
        return _refresh_fn(page, fav_only, m_filter, l_filter, search)

    def _goto_page(delta: int):
        def _fn(page, fav_only, m_filter, l_filter, search):
            return _refresh_fn((int(page or 0)) + delta, fav_only, m_filter, l_filter, search)
        return _fn

    def _reset_to_page_0(page, fav_only, m_filter, l_filter, search):
        # Called when a filter changes.
        return _refresh_fn(0, fav_only, m_filter, l_filter, search)

    filter_inputs = [ui.current_page, ui.favorites_only, ui.model_filter, ui.lora_filter, ui.search_box]
    all_outputs = _render_outputs(ui)

    # Manual refresh — reloads JSON from disk and prints an inventory delta.
    ui.refresh_btn.click(
        fn=_refresh_with_debug, inputs=filter_inputs, outputs=all_outputs, show_progress="hidden",
    )

    # Filter changes reset to page 0
    for comp in (ui.favorites_only, ui.model_filter, ui.lora_filter):
        comp.change(fn=_reset_to_page_0, inputs=filter_inputs, outputs=all_outputs, show_progress="hidden")
    ui.search_box.submit(fn=_reset_to_page_0, inputs=filter_inputs, outputs=all_outputs, show_progress="hidden")

    # Pagination
    ui.prev_btn.click(fn=_goto_page(-1), inputs=filter_inputs, outputs=all_outputs, show_progress="hidden")
    ui.next_btn.click(fn=_goto_page(+1), inputs=filter_inputs, outputs=all_outputs, show_progress="hidden")

    # Auto-refresh when a new generation completes
    if output_trigger is not None:
        output_trigger.change(fn=_refresh_fn, inputs=filter_inputs, outputs=all_outputs, show_progress="hidden")

    # ------------- Per-row events -------------

    def _on_favorite(path, page, fav_only, m_filter, l_filter, search):
        if isinstance(path, str) and path:
            history_persistence.toggle_favorite(path, audio=False)
        return _refresh_fn(page, fav_only, m_filter, l_filter, search)

    def _push_and_select(state_obj, path, exists):
        """Push the history entry into the live top Gallery and return the
        Gallery + last_choice updates. ``exists`` lets us no-op cleanly when
        the underlying file is gone."""
        if not exists or not isinstance(path, str) or not path:
            gr.Info("File is missing from disk; cannot load into the player.")
            return gr.update(), gr.update()
        try:
            gen = get_gen_info(state_obj)
        except Exception as exc:
            _log(f"get_gen_info failed: {exc}")
            return gr.update(), gr.update()
        # Pull the latest persisted settings for this path (authoritative).
        settings = None
        for e in history_persistence.snapshot_entries(audio=False):
            if e.get("path") == path:
                settings = e.get("settings")
                break
        idx = push_to_top_gallery(gen, path, settings)
        file_list = gen.get("file_list") or []
        return gr.Gallery(value=file_list, selected_index=idx), idx

    def _on_play(state_obj, path, exists):
        g, lc = _push_and_select(state_obj, path, exists)
        return g, lc

    def _on_action_with_trigger(state_obj, path, exists):
        """Shared implementation for Extract / To Source / To Control: push to
        top Gallery, update last_choice, and emit a fresh trigger value so
        wgp.py's existing chain fires."""
        g, lc = _push_and_select(state_obj, path, exists)
        trig = str(time.time()) if exists else gr.update()
        return g, lc, trig

    def _delete_row_action(row_idx: int):
        """Factory: returns a click handler for the delete button on row
        ``row_idx``. Keeps the full-render behavior on fire; keeps a targeted
        update on arm."""
        def _fn(state_obj, path, armed_ts, page, fav_only, m_filter, l_filter, search):
            now = time.time()
            if isinstance(armed_ts, (int, float)) and armed_ts > 0 and (now - armed_ts) < DELETE_CONFIRM_WINDOW_S:
                # Fire: delete then full refresh.
                if isinstance(path, str) and path:
                    def _thumb_cb(p):
                        video_thumbs.delete_thumbnail_for(p, ui.save_dir)
                    history_persistence.delete_entry(path, audio=False, delete_file=True, thumb_cb=_thumb_cb)
                    try:
                        gen = get_gen_info(state_obj)
                        fl = gen.get("file_list") or []
                        fsl = gen.get("file_settings_list") or []
                        if path in fl:
                            idx = fl.index(path)
                            fl.pop(idx)
                            if idx < len(fsl):
                                fsl.pop(idx)
                            sel = gen.get("selected", 0)
                            if isinstance(sel, int) and sel >= len(fl):
                                gen["selected"] = max(len(fl) - 1, 0)
                    except Exception as exc:
                        _log(f"top-gallery cleanup after delete failed: {exc}")
                gallery_update = _current_top_gallery_update(state_obj)
                refresh_updates = _refresh_fn(page, fav_only, m_filter, l_filter, search)
                return (gallery_update, *refresh_updates)
            # Arm: small partial update to this row only.
            n_cols = len(_render_outputs(ui))
            per_row = 14
            targets_per_row = list(range(row_idx * per_row, row_idx * per_row + per_row))
            # Build a tuple of gr.update() for all outputs, then override this row's
            # delete button + cancel button + armed state.
            base = [gr.update()] * n_cols
            base[targets_per_row[9]] = gr.update(  # delete_btn → red Confirm
                value=DELETE_CONFIRM_LABEL,
                variant="stop",
                elem_classes=[_DELETE_BASE_CLASS, _DELETE_ARMED_CLASS],
            )
            base[targets_per_row[10]] = gr.update(visible=True)  # cancel_btn appears
            base[targets_per_row[13]] = now                       # delete_armed_state
            return (gr.update(), *base)
        return _fn

    def _cancel_delete_action(row_idx: int):
        """Return a click handler that disarms a single row without firing a
        full page render. Returns updates only for this row's delete + cancel
        + armed_state."""
        def _fn():
            return [
                gr.update(value=DELETE_LABEL, variant="secondary", elem_classes=[_DELETE_BASE_CLASS]),  # delete_btn
                gr.update(visible=False),  # cancel_btn
                0.0,                       # delete_armed_state
            ]
        return _fn

    def _current_top_gallery_update(state_obj):
        try:
            gen = get_gen_info(state_obj)
            fl = gen.get("file_list") or []
            sel = gen.get("selected", 0) if fl else None
            return gr.Gallery(value=fl, selected_index=sel)
        except Exception:
            return gr.update()

    # Wire each row
    for row_idx, r in enumerate(ui.rows):
        # Favorite toggle
        r.fav_btn.click(
            fn=_on_favorite,
            inputs=[r.path_state, *filter_inputs],
            outputs=all_outputs,
            show_progress="hidden",
        )
        # Play (push to top Gallery)
        r.play_btn.click(
            fn=_on_play,
            inputs=[state, r.path_state, r.exists_state],
            outputs=[top_gallery, last_choice],
            show_progress="hidden",
        )
        # Extract Settings — emit hidden trigger
        r.extract_btn.click(
            fn=_on_action_with_trigger,
            inputs=[state, r.path_state, r.exists_state],
            outputs=[top_gallery, last_choice, ui.extract_trigger],
            show_progress="hidden",
        )
        # To Video Source
        r.to_source_btn.click(
            fn=_on_action_with_trigger,
            inputs=[state, r.path_state, r.exists_state],
            outputs=[top_gallery, last_choice, ui.to_source_trigger],
            show_progress="hidden",
        )
        # To Control Video
        r.to_control_btn.click(
            fn=_on_action_with_trigger,
            inputs=[state, r.path_state, r.exists_state],
            outputs=[top_gallery, last_choice, ui.to_control_trigger],
            show_progress="hidden",
        )
        # Delete: handler factory includes the row index
        r.delete_btn.click(
            fn=_delete_row_action(row_idx),
            inputs=[state, r.path_state, r.delete_armed_state, *filter_inputs],
            outputs=[top_gallery, *all_outputs],
            show_progress="hidden",
        )
        # Cancel: dismiss the armed state for this row only.
        r.cancel_btn.click(
            fn=_cancel_delete_action(row_idx),
            inputs=[],
            outputs=[r.delete_btn, r.cancel_btn, r.delete_armed_state],
            show_progress="hidden",
        )


def initial_render(ui: HistoryBrowserUI):
    """Render the first page at app load (and populate filter choices)."""
    return _render_page(ui, 0, False, [], [], "")


def initial_render_outputs(ui: HistoryBrowserUI) -> list[Any]:
    return _render_outputs(ui)
