---
name: decoy-decode
description: Decode decoy-font images that hide the real message behind sharp fake outlines using spatial-frequency separation and optional OCR. Use for hybrid-image text, gibberish letters with soft blurry shadows, text that changes when you squint or view from far away, or requests asking what a decoy-font image really says.
---

# Decoy-Font Image Decoder

Decoy-font images pack **two messages into the same glyphs** at different
spatial frequencies: a sharp thin-outline letter (the decoy — what any naive
read or OCR returns) and a heavily blurred second letter behind it (the real
message — what humans see when they squint). This skill separates the layers
into **two images**, then reads both messages from them.

## Hard rules — ONE run producing TWO images

The #1 failure of this skill is trusting a naive read: the sharp outlines are a
trap planted specifically for AI, and reading the raw image reports the fake
message with full confidence. The #2 failure is over-processing — spawning
diagnostic images and hallucinating text. Do neither.

- **Never report text read directly from the raw image.** That is the decoy
  layer by design. The real message only exists at low spatial frequency.
- **Run the decoder exactly once.** The algorithm below is correct and
  complete. Do not write a second decoder, try another method (edge detection,
  adaptive thresholding, frequency-domain analysis, per-letter crops…), or
  "improve" the pipeline.
- **Produce exactly two images: `revealed.png` and `decoy.png`.** Create NO
  other images — no diagnostic maps, crops, or re-thresholded variants.
- **Read the two images, then stop.** Soft, rounded, blobby letters in
  `revealed.png` are the normal, correct output. If you can read the words,
  report them.

## Workflow

1. **Resolve the image** from the user's request, or the most recently modified
   `.png`/`.jpg`/`.jpeg`/`.webp` in the working directory. Ask only if several
   candidates are plausible.
2. **Check the runtime:** `python -c "import cv2, numpy"`. If imports fail,
   install `opencv-python-headless` and `numpy` (from
   `<plugin-root>/requirements.txt` when present), asking first only if the
   environment requires approval.
3. **Decode — run once.** Pick ONE:
   - If `<plugin-root>/decode.py` exists:
     `python "<plugin-root>/decode.py" "<image>" -o "<out-dir>"`
   - Otherwise (plugin files not present in this environment): write the
     **Decoder** program at the bottom of this file verbatim to a scratch
     `decode.py`, then `python decode.py "<image>" "<out-dir>"`.

   Either path writes exactly `revealed.png` and `decoy.png`. Determine
   `<plugin-root>` from this skill's installed location
   (`<plugin-root>/codex-skills/decoy-decode/SKILL.md`), not the working
   directory.
4. **Read both messages.** Read `revealed.png` — dark, soft letters on white;
   that is the REAL message. Read `decoy.png` for the fake sharp-outline
   message. Any printed `Hidden text` line is only a rough OCR hint — trust
   your own reading of the images. Mark any single ambiguous glyph
   `(unclear: X)`.

## Required chat response

Render **both** images, then state the two texts — nothing else:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

![decoy.png](<absolute-path-to-decoy.png>)

Hidden text in the image: **<REAL TEXT>**
Decoy text (the fake layer AI reads): **<FAKE TEXT>**
```

Use absolute local paths so Codex renders the images in chat. Do not claim
success if the program did not run or you did not inspect `revealed.png`. If
`revealed.png` has no letter shapes (just a uniform smudge), say no hidden text
was recovered — still show the two images.

## Troubleshooting (still one run, still two images)

- **Washed-out or merged reveal:** rerun the SAME decoder once with
  `--sigma-frac 0.015`. That is the only permitted retry.
- **No Tesseract:** fine — read the text from `revealed.png` yourself.

## Decoder (write to a scratch `decode.py` only if the bundled one is absent)

```python
import sys, os
import cv2, numpy as np

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "image.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
if img is None:
    sys.exit(f"cannot open image: {IMAGE}")

# --- split the two spatial-frequency layers ---
inv = 255 - img.astype(np.float32)          # text mass -> bright
sigma = max(img.shape) * 0.01               # ~1% of the long edge kills thin outlines
low = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)

# hidden message: only the blurred mass survives the low-pass filter
norm = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX)
revealed = (255 - np.power(norm / 255.0, 0.6) * 255).astype(np.uint8)

# decoy message: what remains after removing the low-frequency mass
high = np.clip(inv - low, 0, None)
high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
decoy = (255 - high).astype(np.uint8)

cv2.imwrite(os.path.join(OUT, "revealed.png"), revealed)
cv2.imwrite(os.path.join(OUT, "decoy.png"), decoy)
print("done — wrote revealed.png (hidden text) and decoy.png (fake sharp layer)")
```
