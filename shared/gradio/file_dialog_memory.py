"""Separate "last used folder" memory for audio and image/video file dialogs.

A plain ``<input type="file">`` dialog opens wherever the browser last left off,
one shared folder for every upload on every page, and a web page cannot choose
it. Chromium's File System Access picker (``showOpenFilePicker``) opens the same
native Explorer/Finder dialog but remembers a separate last folder per ``id``.

This script intercepts the file dialogs Gradio opens (it calls ``.click()`` on a
hidden file input) and, for audio inputs and for image/video inputs, opens the
picker with its own id instead, then hands the chosen files back to the input
exactly as a normal selection would. Anything else — other file types, browsers
without the API (Safari, Firefox), or non-secure origins such as plain http on a
LAN IP — falls through to the browser's normal dialog unchanged.
"""

AUDIO_PICKER_ID = "wangp-audio"
MEDIA_PICKER_ID = "wangp-media"


def get_javascript() -> str:
    return r"""
    (function () {
        if (window.__wangpFileDialogMemory) return;
        window.__wangpFileDialogMemory = true;
        if (typeof window.showOpenFilePicker !== "function" || !window.isSecureContext) return;

        const EXTS = {
            audio: [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".webm"],
            image: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".pjpeg", ".avif"],
            video: [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ogv"],
        };
        const kindOfExt = (ext) => Object.keys(EXTS).find((k) => EXTS[k].includes(ext));

        // Map an input's accept list to picker types grouped by kind.
        function parseAccept(input) {
            const kinds = { audio: new Set(), image: new Set(), video: new Set() };
            const tokens = String(input.accept || "").split(",").map((t) => t.trim().toLowerCase()).filter(Boolean);
            for (const t of tokens) {
                const wildcard = t.match(/^(audio|image|video)\/\*$/);
                if (wildcard) { EXTS[wildcard[1]].forEach((e) => kinds[wildcard[1]].add(e)); continue; }
                const mime = t.match(/^(audio|image|video)\//);
                if (mime) { EXTS[mime[1]].forEach((e) => kinds[mime[1]].add(e)); continue; }
                const ext = t.startsWith(".") ? t : null;
                const kind = ext && kindOfExt(ext);
                if (kind) kinds[kind].add(ext);
            }
            return kinds;
        }

        function pickerIdFor(input) {
            if (input.type !== "file" || input.webkitdirectory) return null;
            const kinds = parseAccept(input);
            const insideAudio = !!input.closest("[data-testid='audio'], .gradio-audio");
            if (kinds.audio.size && !kinds.image.size && !kinds.video.size) return "AUDIO_ID";
            if (insideAudio && !kinds.image.size && !kinds.video.size) return "AUDIO_ID";
            if ((kinds.image.size || kinds.video.size) && !kinds.audio.size) return "MEDIA_ID";
            return null;  // mixed, untyped or other files: leave the native dialog alone
        }

        function pickerTypes(input, id) {
            const kinds = parseAccept(input);
            if (id === "AUDIO_ID" && !kinds.audio.size) EXTS.audio.forEach((e) => kinds.audio.add(e));
            const labels = { audio: "Audio", image: "Images", video: "Videos" };
            return Object.keys(kinds).filter((k) => kinds[k].size)
                .map((k) => ({ description: labels[k], accept: { [k + "/*"]: [...kinds[k]] } }));
        }

        async function openWithMemory(input, id) {
            let handles;
            try {
                handles = await window.showOpenFilePicker({ id, multiple: !!input.multiple, types: pickerTypes(input, id), excludeAcceptAllOption: false });
            } catch (err) {
                if (err && err.name === "AbortError") return;      // user cancelled
                console.warn("[WanGP] picker unavailable, using the normal dialog:", err);
                nativeClick.call(input);
                return;
            }
            const transfer = new DataTransfer();
            for (const handle of handles) transfer.items.add(await handle.getFile());
            input.files = transfer.files;
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        }

        const nativeClick = HTMLInputElement.prototype.click;
        HTMLInputElement.prototype.click = function () {
            const id = pickerIdFor(this);
            if (!id) return nativeClick.call(this);
            openWithMemory(this, id === "AUDIO_ID" ? "__AUDIO__" : "__MEDIA__");
        };
        const nativeShowPicker = HTMLInputElement.prototype.showPicker;
        if (typeof nativeShowPicker === "function") {
            HTMLInputElement.prototype.showPicker = function () {
                const id = pickerIdFor(this);
                if (!id) return nativeShowPicker.call(this);
                openWithMemory(this, id === "AUDIO_ID" ? "__AUDIO__" : "__MEDIA__");
            };
        }
        console.log("[WanGP] separate last-folder memory for audio and image/video dialogs is active");
    })();
    """.replace("__AUDIO__", AUDIO_PICKER_ID).replace("__MEDIA__", MEDIA_PICKER_ID)
