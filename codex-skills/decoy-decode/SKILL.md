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
- **Read the hidden text directly off `revealed.png`, then stop.** Report
  exactly what you can read. If one glyph won't resolve, mark it
  `(unclear: X)`; if the message won't resolve at all, say so — never fill
  in a guess.

## Workflow

1. **Resolve the image** from the user's request, or the most recently modified
   `.png`/`.jpg`/`.jpeg`/`.webp` in the working directory. Ask only if several
   candidates are plausible.
2. **Run everything inside your own computing sandbox.** All commands —
   dependency checks, installs, the decoder run — execute in the sandboxed
   environment Codex provides for this task, never against the user's system
   runtime. Check the sandbox runtime with
   `python -c "import cv2, numpy; cv2.imread"` (this also catches a broken
   OpenCV install whose `cv2` imports but has no functions — e.g. an OpenCV 5
   wheel with no `imread`). If it fails, do NOT touch the user's system
   Python or its site-packages, do NOT ask for permission, and do NOT create
   a virtualenv or "task-local environment" — install into a sandbox
   workspace folder and point Python at it:

   - `pip install --quiet --target ".codex-runtime-deps" "opencv-python-headless<5" numpy`
   - run every later `python` command with that folder on `PYTHONPATH`
     (POSIX: `PYTHONPATH=".codex-runtime-deps" python …`; Windows PowerShell:
     `$env:PYTHONPATH=".codex-runtime-deps"; python …`).

   OpenCV must stay on 4.x — the 5.x wheels can expose an incomplete `cv2`.
   This one method covers every environment: it needs no elevated rights, it
   works when the system site-packages directory is owned by an elevated
   installer and unreadable to the task (a real failure mode on Windows), and
   because `PYTHONPATH` precedes site-packages, the pinned 4.x copy shadows
   any broken system OpenCV. Do NOT install from `requirements.txt` (it lists
   `pytesseract`, which this skill never needs — you read the output images
   with vision, not OCR). If installs are blocked entirely, state that
   one-line limitation and stop.
3. **Decode — run once.** Pick ONE:
   - If `<plugin-root>/decode.py` exists:
     `python "<plugin-root>/decode.py" "<image>" -o "<out-dir>" --no-ocr`
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

Use absolute local paths so Codex renders the images in chat. Keep the reply
to exactly this — no environment-setup narration, no dependency logs, no
intermediate steps. Do not claim
success if the program did not run or you did not inspect `revealed.png`. If
`revealed.png` has no letter shapes (just a uniform smudge), say no hidden text
was recovered — still show the two images.

## Troubleshooting (still one run, still two images)

- **Washed-out or merged reveal:** rerun the SAME decoder once with
  `--sigma-frac 0.0075`. That is the only permitted retry.
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
