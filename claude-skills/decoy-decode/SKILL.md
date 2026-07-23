---
name: decoy-decode
description: Decode decoy-font or hybrid-image text into revealed.png, present that generated image, and transcribe the text visible in it. Use when the user asks what a decoy-font image says, mentions blurry letterforms or text that changes when squinting, or requests the revealed layer.
---

# Decoy-Font Image Decoder

Decode the image once, present `revealed.png`, then perform a plain visual
transcription of the presented image.

## Hard rules

- Present `revealed.png` before attempting to read it.
- After presentation completes, open `revealed.png` with the available image
  viewer and answer: `What text is visible in this image?`
- **Run the decoder exactly once.** The algorithm below is correct and
  complete. Do not write a second decoder, try another method (edge detection,
  adaptive thresholding, frequency-domain analysis, per-letter crops…), or
  "improve" the pipeline.
- **Produce exactly two images: `revealed.png` and `decoy.png`.** Create NO
  other images — no diagnostic maps, no crops, no re-thresholded variants.
- Transcribe only from `revealed.png`, not the source image, `decoy.png`, OCR,
  terminal output, word counts, or candidate phrases.

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
     `python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<image>" -o "<out-dir>" --no-ocr`
   - Otherwise (plugin files not present in this environment): write the
     **Decoder** program at the bottom of this file verbatim to a scratch
     `decode.py`, then `python decode.py "<image>" "<out-dir>"`.

   Either path writes exactly `revealed.png` and `decoy.png` and nothing else.
   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install dir (on Windows PowerShell,
   `$env:CLAUDE_PLUGIN_ROOT`); if it expands empty, use the embedded Decoder
   instead.
4. **Present, then read.** Present `revealed.png` with the platform's file tool
   and wait for it to finish. Then open that same file with the image viewer
   and transcribe the visible text.

## Required response format

Reply with:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

Text in revealed.png: **<LITERAL VISUAL TRANSCRIPTION>**
```

Use an absolute local path so the image renders in chat.

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

# crop/stack and enlarge, so the letters stay big and readable even after
# chat-UI downscaling; the revealed layer is split into its text lines and
# stacked tightly so every line fills the frame at maximum size
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

def stack_lines(layer, ref, gap=40):
    rows = (ref > 127).any(axis=1)
    bands, in_band, start = [], False, 0
    for i, r in enumerate(rows):
        if r and not in_band:
            start, in_band = i, True
        elif not r and in_band:
            bands.append((start, i)); in_band = False
    if in_band:
        bands.append((start, len(rows)))
    bands = [b for b in bands if b[1] - b[0] > 10]
    if len(bands) <= 1:
        return layer
    pieces = []
    for y0, y1 in bands:
        xs = np.where((ref[y0:y1] > 127).any(axis=0))[0]
        pieces.append(layer[max(0, y0 - gap // 2):y1 + gap // 2,
                            max(0, int(xs.min()) - gap // 2):int(xs.max()) + gap // 2])
    canvas = np.zeros((sum(p.shape[0] for p in pieces) + gap * (len(pieces) - 1),
                       max(p.shape[1] for p in pieces)), dtype=layer.dtype)
    y = 0
    for p in pieces:
        canvas[y:y + p.shape[0], :p.shape[1]] = p
        y += p.shape[0] + gap
    return canvas

stacked = stack_lines(revealed, norm)
f = 1600 / max(stacked.shape)
if f > 1:
    stacked = cv2.resize(stacked, (int(stacked.shape[1] * f), int(stacked.shape[0] * f)),
                         interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT, "revealed.png"), 255 - stacked)
cv2.imwrite(os.path.join(OUT, "decoy.png"), 255 - crop_to_text(high, norm))
print("done — wrote revealed.png (hidden text) and decoy.png (fake sharp layer)")
```
