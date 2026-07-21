---
name: decoy-decode
description: Use when an image is written in a "decoy font" — hybrid-image text where sharp outline letters spell one (fake) message and blurry blobs behind them hide the real one, gibberish letters with soft shadows, text that changes when you squint or step back, or the user asks what a decoy-font image really says.
---

# Decoy-Font Image Decoder

A decoy-font image packs two messages into the same glyphs: a sharp thin-outline letter (the decoy — what any naive read returns) and a heavily blurred second letter behind it (the real message — what humans see when they squint). The decoder below separates the two spatial-frequency layers into two images. Run it once, read both messages from the images, and report them. Nothing more.

## The whole job — do exactly this

1. Write the program in **Decoder** (below) to `decode.py`, then run it once on the attached image:

   ```bash
   pip install --quiet opencv-python-headless numpy
   python decode.py "<image-path>" out
   ```

2. View **both** `out/revealed.png` (the real hidden message — soft dark letters on white) and `out/decoy.png` (the fake sharp-outline message). Read the word(s) directly from these images.

3. Reply with **only** this — show both images, then the two texts:

   ```
   ![revealed.png](out/revealed.png)

   ![decoy.png](out/decoy.png)

   Hidden text in the image: **<REAL TEXT>**
   Decoy text (the fake layer AI reads): **<FAKE TEXT>**
   ```

## Trust the images — do NOT over-process

The hidden letters are soft, rounded blobs. **That is the correct, finished output — read it as-is.** The most common failure here is reporting the sharp text from the raw image (that is the planted fake message) or not trusting a perfectly readable reveal and doing pointless extra work that ends in a hallucinated answer. So:

- **Never report text read directly from the raw image.** The sharp outlines are a decoy aimed specifically at AI; the real message exists only at low spatial frequency.
- **Run the decoder once.** Do not build a second decoder, try another method (edge detection, adaptive thresholding, frequency-domain analysis), or "improve" the approach. One run, then read.
- **Produce exactly two images** — `revealed.png` and `decoy.png`. Do not create any other images: no crops, no diagnostic maps, no re-thresholded variants.
- **Never conclude "it's just a smudge" or "the decode failed"** because the letters look soft and blobby. Soft blobby letters = success. Look for the words.
- If one character is genuinely ambiguous, read the rest and mark just that one `(unclear: X)`. Keep the reply to the two images plus the two text lines.

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
