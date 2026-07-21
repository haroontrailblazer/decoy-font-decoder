"""Decode decoy-font images (mixfont.com "Decoy Font" hybrid-image trick).

Each glyph hides two letters at different spatial frequencies: a sharp
thin-outline decoy letter and a heavily blurred real letter behind it.
A strong low-pass filter (the computational equivalent of squinting)
erases the outlines, leaving only the hidden message.

Usage:
  python squint_decode.py                # decode every .png in this folder
  python squint_decode.py image.png ...  # decode specific images
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "decoded")


def decode(path, out_dir=OUT_DIR):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("skip (not an image):", path)
        return None
    inv = 255 - img.astype(np.float32)  # text mass -> bright
    # sigma scaled to image size: ~1% of the long edge kills thin outlines
    sigma = max(img.shape) * 0.01
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    boosted = np.power(norm / 255.0, 0.6) * 255  # darken letter mass
    out = (255 - boosted).astype(np.uint8)  # dark text on white
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0] + "_decoded.png"
    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, out)
    print("wrote", out_path)
    return out_path


def main():
    paths = sys.argv[1:] or [
        p for p in glob.glob(os.path.join(HERE, "*.png"))
        if not p.endswith("_decoded.png")
    ]
    if not paths:
        sys.exit("no .png files found to decode")
    for path in paths:
        decode(path)


if __name__ == "__main__":
    main()
