---
description: Decode a decoy-font image and reveal the real hidden text
argument-hint: [image-path]
---

Immediately decode a decoy-font image (two messages hidden at different
spatial frequencies — sharp fake outlines over blurred real letters) by
running the bundled decoder. Do the work now — do NOT ask the user to set up a
folder, install the plugin differently, or point at files beyond the image path.

Target image: $ARGUMENTS

If `$ARGUMENTS` is empty, use the most recently modified `.png`, `.jpg`,
`.jpeg`, or `.webp` file in the current directory. Only ask the user if there
is more than one plausible candidate.

Then, without further prompting:

1. Check deps: `python -c "import cv2, numpy; cv2.imread"` (also catches a
   broken OpenCV whose `cv2` imports but has no functions). If it fails, run
   `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"`.
2. Run the decoder (it writes `revealed.png` and `decoy.png`):

   ```
   python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<image-path>" -o "<scratch-out-dir>" --no-ocr
   ```

   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install directory. On Windows
   PowerShell it is `$env:CLAUDE_PLUGIN_ROOT`; if it expands empty, substitute
   the install path literally.
3. If running on claude.ai (`/mnt/user-data` paths, or an empty plugin root
   with the skill under `/mnt/skills/plugins/`), do not call `view`, OCR, or
   transcribe the generated image. Present only `revealed.png` and reply:

   ```
   Attach revealed.png in your next message and ask: What text is in this image?
   ```

4. If running in Claude Code, open `revealed.png` once with the local image
   reader and return `Text in revealed.png: **<LITERAL TRANSCRIPTION>**`.

Never inspect `decoy.png`. On claude.ai, never guess the text from the
generated-file preview; the fresh user attachment is the required second step.
