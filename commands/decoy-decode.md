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
   python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<image-path>" -o "<scratch-out-dir>"
   ```

   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install directory. On Windows
   PowerShell it is `$env:CLAUDE_PLUGIN_ROOT`; if it expands empty, substitute
   the install path literally.
3. Read `revealed.png` with vision to get the REAL message (soft dark letters
   on white). Read `decoy.png` to get the fake sharp-outline message.
4. Reply with both texts, plus the output image paths:

   ```
   Hidden text in the image: **<REAL TEXT>**
   Decoy text (the fake layer AI reads): **<FAKE TEXT>**
   ```

Never report text read directly off the raw image — the sharp outlines are a
decoy planted for AI; the real message exists only in the blurred low-frequency
layer. Never reconstruct the hidden text from word lengths, theme, or
plausibility — the decoy shares its per-word letter counts with the hidden
message, so a phrase "fitting the lengths" is zero evidence; read the rendered
pixels or say you could not. Do not claim success unless the decoder actually
ran and you inspected `revealed.png`.
