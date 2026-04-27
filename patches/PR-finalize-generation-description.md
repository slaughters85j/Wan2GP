# Preserve Gallery value on finalize_generation

## Problem

`finalize_generation` fires at the end of the queue-resume chain that runs on `main.load`. Its return value for the output Gallery is currently:

```python
gr.Gallery(selected_index=choice)
```

In Gradio 4, returning a freshly-constructed Component from an event handler is treated as a **full** update — fields not explicitly passed fall back to the component's defaults. For `gr.Gallery` the default `value` is `None`, so this return wipes whatever the gallery is currently showing and replaces it with an empty state.

Today this is harmless because `gen["file_list"]` is reliably empty at launch (the load chain hits `process_tasks` → early-return → `finalize_generation` before anything has populated it), so the reset is "empty → empty."

It stops being harmless as soon as something pre-populates `gen["file_list"]` before `finalize_generation` runs — for example, a persistence layer that seeds the gallery from the last session's outputs, or any downstream plugin / extension that stages items into the gallery at build time. Those items briefly render, then disappear ~200ms into page load.

## Reproduction

1. Launch the app normally.
2. Before the browser fully loads, populate `gen["file_list"]` with one or more file paths (e.g. via a persistence seed in `generate_video_tab` after `state_dict["gen"] = gen`).
3. Pass the same list as `value=` to the `gr.Gallery` at build time.
4. Observe the gallery renders with the items, then clears when `finalize_generation` fires from the queue chain.

## Fix

Pass the current `file_list` as `value=` so the return is a correct re-assertion rather than an unintended reset:

```diff
-    return gallery_tabs, 1 if last_was_audio else 0, gr.update() if last_was_audio else gr.Gallery(selected_index=choice),  *pack_audio_gallery_state(audio_file_list, audio_choice), ...
+    return gallery_tabs, 1 if last_was_audio else 0, gr.update() if last_was_audio else gr.Gallery(value=gen.get("file_list", []), selected_index=choice),  *pack_audio_gallery_state(audio_file_list, audio_choice), ...
```

One line, single file (`wgp.py`).

## Risk

Minimal. The existing value in `gen["file_list"]` is exactly what the gallery should be showing at this point — `finalize_generation` already reads it four lines above to compute `choice`. Passing it explicitly as `value=` is a no-op for all existing call sites (gallery was either already displaying that list or about to, so the state is identical).

The `audio_last_selected` branch is unchanged (`gr.update()`), so audio behavior is untouched.

## Why this matters even without the triggering scenario

Independent of any specific use case, returning a partially-constructed `gr.Gallery` from an event handler while relying on the frontend to preserve fields you didn't set is a Gradio 4 footgun that resurfaces whenever the component's state is non-empty. Making the return explicit is defensive regardless of whether the current codebase has a triggering scenario.
