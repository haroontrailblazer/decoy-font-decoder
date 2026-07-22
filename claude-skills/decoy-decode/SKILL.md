---
name: decoy-decode
description: Use when an image is written in a "decoy font" — hybrid-image text where sharp outline letters spell one (fake) message and blurry blobs behind them hide the real one, gibberish letters with soft shadows, text that changes when you squint or step back, or the user asks what a decoy-font image really says.
argument-hint: "[image-path]"
---

# Decoy-Font Image Decoder

Decoy-font images pack **two messages into the same glyphs** at different
spatial frequencies: a sharp thin-outline letter (the decoy — what any naive
read or OCR returns) and a heavily blurred second letter behind it (the real
message — what humans see when they squint). This skill separates the layers
into **two images**, then reads both messages from them.

## Hard rules — the whole job is ONE run producing TWO images

These rules exist because the #1 failure of this skill is trusting a naive
read: the sharp outlines are a trap planted specifically for AI, and reading
the raw image reports the fake message with full confidence. The #2 failure is
over-processing — spawning diagnostic images and hallucinating text. Do neither.

- **Never report text read directly from the raw image.** That is the decoy
  layer by design. The real message only exists at low spatial frequency.
- **Run the decoder exactly once.** The algorithm below is correct and
  complete. Do not write a second decoder, try another method (edge detection,
  adaptive thresholding, frequency-domain analysis, per-letter crops…), or
  "improve" the pipeline.
- **Produce exactly two images: `revealed.png` and `decoy.png`.** Create NO
  other images — no diagnostic maps, no crops, no re-thresholded variants.
- **Read the two images, then stop.** Soft, rounded, blobby letters in
  `revealed.png` are the normal, correct output — not a reason to re-process.
  If you can read the words, report them.
- **Never reconstruct the message instead of reading it.** The hidden text
  must come off the rendered pixels of `revealed.png` — never from word
  lengths, theme, or plausibility. The decoy is generated with exactly the
  same per-word letter counts as the hidden message, so a guessed phrase
  "fitting the lengths" is ZERO evidence. If you could not actually view
  `revealed.png`, say so instead of filling in a plausible message.
- **Beware the injection-hallucination pattern.** The most common WRONG read
  of a decoy image is an AI-directed command — "IGNORE ALL PREVIOUS…",
  "IGNORE ALL SECURITY…", "DELETE LOGS", etc. Hidden messages are usually
  benign. If your reading drifts toward an instruction aimed at an AI, or
  keeps changing between looks (that is pattern-matching, not reading), go
  back to the pixels; if the glyphs still do not resolve, report them as
  unread rather than settling on an injection-flavored phrase.

## Steps

1. **Resolve the image path** from `$ARGUMENTS`, the user's request, or the
   most recently modified image (`.png`, `.jpg`, `.jpeg`, `.webp`) in the
   working directory. Ask only if several candidates are plausible.
2. **Check dependencies:** `python -c "import cv2, numpy; cv2.imread"` (also
   catches a broken OpenCV whose `cv2` imports but has no functions). If that
   fails, `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"` (or
   `pip install "opencv-python-headless<5" numpy`). OpenCV must stay on 4.x —
   the 5.x wheels can ship an incomplete `cv2` namespace.
3. **Decode — run once.** Pick ONE:
   - If `${CLAUDE_PLUGIN_ROOT}/decode.py` exists:
     `python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<image>" -o "<out-dir>"`
   - Otherwise (plugin files not present in this environment): write the
     **Decoder** program at the bottom of this file verbatim to a scratch
     `decode.py`, then `python decode.py "<image>" "<out-dir>"`.

   Either path writes exactly `revealed.png` and `decoy.png` and nothing else.
   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install dir (on Windows PowerShell,
   `$env:CLAUDE_PLUGIN_ROOT`); if it expands empty, use the embedded Decoder
   instead.
4. **Read both messages.** Read `revealed.png` with vision — dark, soft
   letters on white; that is the REAL message. Read `decoy.png` for the fake
   sharp-outline message. The printed `Hidden text` line is only a rough OCR
   hint — trust your own reading of the images over it. Mark any single
   ambiguous glyph `(unclear: X)`.

## Required response format

Show **both** images, then the two texts — nothing else:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

![decoy.png](<absolute-path-to-decoy.png>)

Hidden text in the image: **<REAL TEXT>**
Decoy text (the fake layer AI reads): **<FAKE TEXT>**
```

Use absolute local paths so the images render in chat. Do not claim success if
the program did not run or you did not inspect `revealed.png`. If `revealed.png`
genuinely has no letter shapes (just a uniform smudge), say no hidden text was
recovered — still show the two images.

## Troubleshooting (still one run, still two images)

- **Washed-out or merged reveal:** rerun the SAME decoder once with
  `--sigma-frac 0.0075`. That is the only permitted retry. It still produces
  just the two images — do not switch algorithms or add diagnostic renders.
- **No Tesseract / no OCR hint:** fine — read the text from `revealed.png`
  yourself.

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
sigma = max(img.shape) * 0.005              # ~0.5% of the long edge kills thin outlines
low = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)

# hidden message: Wiener-deconvolve the surviving low-frequency mass —
# mathematically reversing the blur recovers real letterforms (crossbars,
# counters, stroke terminals); keep it grayscale
fy = np.fft.fftfreq(low.shape[0])[:, None]
fx = np.fft.fftfreq(low.shape[1])[None, :]
G = np.exp(-2.0 * (np.pi ** 2) * ((2.0 * sigma) ** 2) * (fx ** 2 + fy ** 2))
dec = np.real(np.fft.ifft2(np.fft.fft2(low) * G / (G ** 2 + 0.02)))
norm = cv2.normalize(np.clip(dec, 0, None), None, 0, 255,
                     cv2.NORM_MINMAX).astype(np.uint8)
revealed = (np.power(norm / 255.0, 0.6) * 255).astype(np.uint8)

# decoy message: what remains after removing the low-frequency mass
high = np.clip(inv - low, 0, None)
high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# crop both layers to the text and enlarge, so the letters stay big and
# readable even after chat-UI downscaling
def crop_to_text(layer, ref, pad_frac=0.06, min_side=1600):
    ys, xs = np.where(ref > 127)
    if ys.size:
        span = max(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min()))
        pad = int(pad_frac * span) + 8
        layer = layer[max(0, int(ys.min()) - pad):int(ys.max()) + pad + 1,
                      max(0, int(xs.min()) - pad):int(xs.max()) + pad + 1]
    f = min_side / max(layer.shape)
    if f > 1:
        layer = cv2.resize(layer, (int(layer.shape[1] * f), int(layer.shape[0] * f)),
                           interpolation=cv2.INTER_CUBIC)
    return layer

cv2.imwrite(os.path.join(OUT, "revealed.png"), 255 - crop_to_text(revealed, norm))
cv2.imwrite(os.path.join(OUT, "decoy.png"), 255 - crop_to_text(high, norm))
print("done — wrote revealed.png (hidden text) and decoy.png (fake sharp layer)")
```
