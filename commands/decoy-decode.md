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
3. Open `revealed.png` exactly once with the image/vision tool. On claude.ai,
   use the computer `view` tool and wait until it returns the actual pixels.
   Do not open or read `decoy.png`; it is only an output file to present.
4. Immediately copy only the letters visibly present in `revealed.png`:

   ```
   Text in revealed.png: **<LITERAL VISUAL TRANSCRIPTION>**
   ```

Do not count letters or words, match word lengths, generate candidate phrases,
compare image layers, infer from meaning, or perform a confirmation pass. If a
glyph cannot be read in the single view, write `[unclear]` in that position.
Never fill a gap with a plausible word.
