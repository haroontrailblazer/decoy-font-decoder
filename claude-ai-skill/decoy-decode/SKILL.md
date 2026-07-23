---
name: decoy-decode
description: Decode decoy-font or hybrid-image text into revealed.png, present that generated image, and transcribe the text visible in it. Use when the user asks what a decoy-font image says, mentions blurry letterforms or text that changes when squinting, or requests the revealed layer.
---

# Decoy-Font Image Decoder

Run the decoder, present `revealed.png`, then read that presented image as a
normal image-transcription task.

## The whole job — do exactly this

1. Write the program in **Decoder** (below) to `decode.py`, then run it once on the attached image, writing the outputs to the user-visible outputs directory:

   ```bash
   pip install --quiet "opencv-python-headless<5" numpy
   python decode.py "<image-path>" /mnt/user-data/outputs
   ```

2. Present `/mnt/user-data/outputs/revealed.png` with the file-presentation
   tool. Wait for that presentation tool call to complete.
3. After presentation completes, open that same `revealed.png` with the
   computer image viewer and answer the plain question: **What text is visible
   in this image?** Read from the pixels, not from the source image or terminal
   output.
4. Reply with only:

   ```
   [revealed.png](computer:///mnt/user-data/outputs/revealed.png)

   Text in revealed.png: **<VISIBLE TEXT>**
   ```

## Rules

- Complete the presentation step before attempting the transcription.
- Transcribe only `revealed.png`; do not use the source image, `decoy.png`,
  terminal output, OCR, word counts, or candidate phrases.
- Run the decoder once and keep the two generated files unchanged.

## Decoder (write to `decode.py`, run once)

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
print("NEXT REQUIRED ACTION: present revealed.png, then view the presented image and transcribe it")
```
