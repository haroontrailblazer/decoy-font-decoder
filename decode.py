"""Break "decoy font" images with spatial-frequency separation.

Decoy-font images hide two messages in the same glyphs: a sharp thin-outline
letter (the decoy — what AI/OCR reads) and a heavily blurred letter behind it
(the real message — what humans see when they squint or step back). See the
original experiment: https://www.mixfont.com/experiments/decoy-font

This decoder recovers both layers:
  1. low-pass filter (heavy Gaussian blur) erases the thin outlines,
     leaving only the blurred mass — the hidden message,
  2. subtracting that low-pass layer leaves only the sharp outlines —
     the decoy message,
  3. contrast-stretch both, save them, OCR the revealed layer (optional).

Usage:
  python decode.py decoy-message.png
  python decode.py image.png -o out_dir --sigma-frac 0.015 --no-ocr
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"error: cannot open image: {path}")
    return img


def split_layers(img, sigma_frac=0.01, gamma=0.6):
    """Separate the hidden (low-frequency) and decoy (high-frequency) layers.

    The hidden letters are pure blur — they survive a heavy low-pass filter.
    The decoy outlines are thin strokes with almost no ink mass — the same
    filter erases them completely. Sigma is scaled to the image size so the
    split works at any resolution.
    """
    inv = 255 - img.astype(np.float32)  # text mass -> bright
    sigma = max(img.shape) * sigma_frac
    low = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)

    # hidden message: keep the low-pass mass, boost contrast, dark-on-white
    norm = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX)
    revealed = (255 - np.power(norm / 255.0, gamma) * 255).astype(np.uint8)

    # decoy message: what remains after removing the low-frequency mass
    high = np.clip(inv - low, 0, None)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    decoy = (255 - high).astype(np.uint8)

    return revealed, decoy


def find_tesseract():
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def run_ocr(revealed):
    try:
        import pytesseract
    except ImportError:
        print("OCR skipped: pytesseract not installed (pip install pytesseract)")
        return None
    exe = find_tesseract()
    if not exe:
        print("OCR skipped: tesseract binary not found (winget install UB-Mannheim.TesseractOCR)")
        return None
    pytesseract.pytesseract.tesseract_cmd = exe
    text = pytesseract.image_to_string(revealed, config="--psm 6").strip()
    return text or None


def main():
    ap = argparse.ArgumentParser(description="Decode decoy-font images (two texts hidden at different spatial frequencies)")
    ap.add_argument("image", help="path to the decoy-font image")
    ap.add_argument("-o", "--out-dir", default=".", help="directory for output images (default: current dir)")
    ap.add_argument("--sigma-frac", type=float, default=0.01,
                    help="low-pass sigma as a fraction of the long edge (default 0.01)")
    ap.add_argument("--gamma", type=float, default=0.6,
                    help="gamma boost for the revealed layer (default 0.6)")
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR step")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Decoding {args.image} (sigma-frac {args.sigma_frac})")
    img = load_gray(args.image)
    revealed, decoy = split_layers(img, args.sigma_frac, args.gamma)

    revealed_path = os.path.join(args.out_dir, "revealed.png")
    decoy_path = os.path.join(args.out_dir, "decoy.png")
    cv2.imwrite(revealed_path, revealed)
    cv2.imwrite(decoy_path, decoy)
    print(f"Wrote {revealed_path} (hidden message) and {decoy_path} (decoy outlines)")

    if not args.no_ocr:
        text = run_ocr(revealed)
        if text:
            single_line = " ".join(text.split())
            print("\nHidden text in the image: " + single_line)
        elif text is not None:
            print("\nHidden text in the image: (none found — check revealed.png visually)")


if __name__ == "__main__":
    main()
