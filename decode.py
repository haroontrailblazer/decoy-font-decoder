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
  3. Wiener-deconvolve the hidden layer to reverse the blur and recover
     real letterforms, gamma-boost, crop both layers to the text and
     enlarge — so the letters stay big and readable even when a chat UI
     downscales the image — then OCR the revealed layer (optional).

Usage:
  python decode.py decoy-message.png
  python decode.py image.png -o out_dir --sigma-frac 0.0075 --no-ocr
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


def split_layers(img, sigma_frac=0.005, gamma=0.6):
    """Separate the hidden (low-frequency) and decoy (high-frequency) layers.

    The hidden letters are pure blur — they survive a heavy low-pass filter.
    The decoy outlines are thin strokes with almost no ink mass — the same
    filter erases them completely. Sigma is scaled to the image size so the
    split works at any resolution. The hidden layer stays grayscale (hard
    thresholds destroy letterforms whose identity lives in the blur
    gradients); both layers are cropped to the text and enlarged so the
    letters survive chat-UI downscaling.
    """
    inv = 255 - img.astype(np.float32)  # text mass -> bright
    sigma = max(img.shape) * sigma_frac
    low = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)

    # Wiener deconvolution: mathematically reverse the (Gaussian) blur to
    # recover real letterforms — crossbars, counters, stroke terminals —
    # instead of merely boosting contrast on soft blobs
    dec = wiener_deconv(low, 2.0 * sigma, K=0.02)
    norm = cv2.normalize(np.clip(dec, 0, None), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    revealed_gray = (np.power(norm / 255.0, gamma) * 255).astype(np.uint8)

    # decoy message: what remains after removing the low-frequency mass
    high = np.clip(inv - low, 0, None)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    stacked = stack_lines(revealed_gray, norm)
    f = 1600 / max(stacked.shape)
    if f > 1:
        stacked = cv2.resize(stacked, (int(stacked.shape[1] * f), int(stacked.shape[0] * f)),
                             interpolation=cv2.INTER_CUBIC)
    revealed = 255 - stacked
    decoy = 255 - crop_to_text(high, norm)
    return revealed, decoy


def stack_lines(layer, ref, gap=40):
    """Split the text into its lines and stack them tightly, so every line
    fills the frame at maximum letter size."""
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


def wiener_deconv(img_f, sigma, K=0.02):
    """Frequency-domain Wiener deconvolution of a Gaussian blur."""
    fy = np.fft.fftfreq(img_f.shape[0])[:, None]
    fx = np.fft.fftfreq(img_f.shape[1])[None, :]
    G = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img_f) * G / (G ** 2 + K)))


def crop_to_text(layer, mask, pad_frac=0.06, min_side=1600):
    """Crop to the text bounding box and enlarge, so letters are big and
    obvious instead of small shapes lost in a mostly-empty frame."""
    ys, xs = np.where(mask > 127)
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
    ap.add_argument("--sigma-frac", type=float, default=0.005,
                    help="low-pass sigma as a fraction of the long edge (default 0.005)")
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR step")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Decoding {args.image} (sigma-frac {args.sigma_frac})")
    img = load_gray(args.image)
    revealed, decoy = split_layers(img, args.sigma_frac)

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
