<div align="center">

# Decoy Font Decoder

### Give AI agents squint vision.

Decode the real text hidden behind “decoy font” images with **Codex**,
**Claude Code**, **Claude.ai**, or **plain Python**.

[![GitHub stars](https://img.shields.io/github/stars/haroontrailblazer/decoy-font-decoder?style=for-the-badge&logo=github&label=Stars)](https://github.com/haroontrailblazer/decoy-font-decoder/stargazers)
[![MIT License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](requirements.txt)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Plugin-D97757?style=for-the-badge)](#claude-code)
[![Codex](https://img.shields.io/badge/Codex-Plugin-111111?style=for-the-badge&logo=openai)](#codex)

**The sharp text is the trap. The message is the blur—and this decoder reads it.**

</div>

---

## The problem

Decoy-font images (see the original
[mixfont.com experiment](https://www.mixfont.com/experiments/decoy-font)) pack
two messages into the same glyphs at different **spatial frequencies**. Each
character is a sharp thin-outline letter — the decoy — drawn over a heavily
blurred second letter — the real message. AI vision models and OCR latch onto
the crisp, high-contrast outlines and confidently report the fake text, while
humans who squint or step back apply a natural low-pass filter and read the
real one.

Decoy Font Decoder applies that low-pass filter computationally, separates the
two layers, and gives the agent both the revealed hidden message and the decoy
layer it can verify against.

## Why it stands out

- **Agent-native** — one `decoy-decode` skill across Codex, Claude Code, and Claude.ai.
- **Actually frequency-aware** — separates the layers instead of trusting the sharp edges.
- **Verifiable output** — produces both the revealed hidden layer and the isolated decoy layer.
- **Reports both messages** — the real text plus the fake text planted for AI.
- **Works without an agent** — the same decoder runs directly from Python.
- **OCR is optional** — visual recovery still works when Tesseract is unavailable.
- **Local by design** — decoding runs on the machine executing the skill.

## Quick start

### Claude Code

Install the plugin, then pass it an image:

```text
/plugin marketplace add haroontrailblazer/decoy-font-decoder
/plugin install decoy-font-decoder@decoy-font-tools
/decoy-decode path/to/decoy-image.png
```

You can also ask naturally:

> What does `decoy-image.png` really say?

### Codex

Add this repository as a plugin source, install **Decoy Font Decoder**, start a
new task, and ask:

> Use `$decoy-decode` to tell me what `decoy-image.png` really says.

Codex runs the bundled decoder, inspects the revealed layer, displays both
outputs in the task, and reports the hidden and decoy texts.

### Claude.ai

Build the uploadable skill package:

```text
python claude-ai-skill/build-zip.py
```

Upload `decoy-decode.zip` under **Customize → Skills**, start a new chat,
attach an image, and ask what it really says. See the
[Claude.ai installation guide](claude-ai-skill/README.md) for the complete
setup.

### Plain Python

```bash
pip install -r requirements.txt
python decode.py decoy-font-message.png
```

The included example decodes to:

```text
Hidden text in the image: THE SENTENCE IS WRITTEN IN DECOY FONT
```

while the sharp decoy layer reads `INL BBMFBMQL LZ NPLYIFH IH QBOQV TDHY` —
the gibberish planted for machines. No plugin installation is required. You
can also paste [`prompts/decode-in-chat.md`](prompts/decode-in-chat.md) into a
supported chat, attach the image, and run the same workflow.

---

## How it works

```text
Image
  ↓
Invert (text mass → bright)
  ↓
Heavy Gaussian low-pass (σ ≈ 1% of the long edge)
  ↓
Thin decoy outlines vanish — blurred letter mass survives
  ↓
Contrast + gamma stretch  →  revealed.png (real message)
  ↓
High-frequency remainder  →  decoy.png (fake message)
  ↓
Optional OCR on the revealed layer
```

Under the hood, the decoder:

1. Inverts the grayscale image so ink mass becomes signal.
2. Applies a Gaussian low-pass filter scaled to the image size — the thin
   outlines carry almost no ink mass and are erased; the blurred glyphs survive.
3. Contrast-stretches and gamma-boosts the surviving mass into readable
   dark-on-white text.
4. Subtracts the low-pass layer from the original to isolate the sharp decoy
   outlines.
5. Runs optional Tesseract OCR on the revealed layer and lets the agent verify
   both images visually.

## Output

Each run writes:

| File | Purpose |
| --- | --- |
| `revealed.png` | The real hidden message — the authoritative layer to read |
| `decoy.png` | The isolated sharp decoy layer — the fake text planted for AI |

Useful command-line options:

```bash
python decode.py IMAGE \
  -o OUT_DIR \
  --sigma-frac 0.0075 \
  --gamma 0.6 \
  --no-ocr
```

## Plugin structure

This repository ships native skill packaging for each supported agent:

| Surface | Integration |
| --- | --- |
| Claude Code | `.claude-plugin/plugin.json`, marketplace metadata, and `claude-skills/decoy-decode/SKILL.md` |
| Claude.ai | Uploadable skill built from `claude-ai-skill/decoy-decode/SKILL.md` |
| Codex | `.codex-plugin/plugin.json`, `codex-skills/decoy-decode/SKILL.md`, and `agents/openai.yaml` |
| Any supported chat | Portable workflow in `prompts/decode-in-chat.md` |
| Standalone | `decode.py` and `requirements.txt` |

## Requirements

- Python 3.8 or newer
- OpenCV
- NumPy
- `pytesseract` for optional OCR

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On Windows, Tesseract can be installed with:

```powershell
winget install UB-Mannheim.TesseractOCR
```

Both output layers are still generated when Tesseract is not installed.

## Repository map

```text
decoy-font-decoder/
├── decode.py                     # Spatial-frequency decoder
├── squint_decode.py              # Minimal batch decoder (hidden layer only)
├── decoy-font-message*.png       # Sample decoy-font images
├── claude-skills/                # Claude Code skill
├── claude-ai-skill/              # Claude.ai uploadable skill source
├── codex-skills/                 # Codex skill and UI metadata
├── commands/                     # Claude Code slash command
└── prompts/                      # Portable in-chat workflow
```

## The core insight

Decoy fonts are presented as text that AI cannot read correctly — the sharp
layer is a trap aimed at models that trust high-frequency detail. The harder
truth is simply that the two messages live at **different spatial
frequencies**, and a Gaussian filter separates them in one step.

This project packages that insight as an agent skill, so the same assistant
that receives the image can run the analysis, inspect both layers, and answer
with the real message — while showing you the fake one it was supposed to fall
for.

## License

Released under the [MIT License](LICENSE).

## Author

Built by **Haroon K M**
([@haroontrailblazer](https://github.com/haroontrailblazer)).
