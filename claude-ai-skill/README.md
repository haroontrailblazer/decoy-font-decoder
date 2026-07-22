# Decoy Font Decoder — skill for claude.ai

A ready-to-upload [Agent Skill](https://support.anthropic.com/en/articles/skills)
for **claude.ai** (the web/desktop app). It decodes decoy-font images — glyphs
that show one sharp fake letter to machines while a blurred real letter hides
behind it — by running a small spatial-frequency filter in claude.ai's code
sandbox.

## Build the upload archive

The zip is **not** committed to the repo on purpose — a zip inside the repo is a
nested zip inside the plugin package, and claude.ai rejects that on install
(`Nested zip files are not allowed`). Build it locally in one step:

```bash
python claude-ai-skill/build-zip.py
```

This writes `claude-ai-skill/decoy-decode.zip` containing only
`decoy-decode/SKILL.md` (the decoder is embedded in it).

## Install in claude.ai (2 minutes)

1. In claude.ai, make sure **Code execution and file creation** is enabled,
   then open **Customize → Skills**.
2. Click **Upload skill** and choose the single **`decoy-decode.zip`** you built.
   Upload only that file — do **not** upload a zip of the whole repo or of the
   `claude-ai-skill/` folder, or you'll hit the nested-zip error.
3. Start a new chat, attach a decoy-font image, and ask:

   > What does this image really say?

Claude loads the `decoy-decode` skill, runs the bundled decoder, views
`revealed.png` once, and literally transcribes only the letters visible there.

After replacing an older copy of this skill, start a **new chat** so the old
instructions are not retained in the conversation. During a successful run,
Claude's activity should show one `view` operation for `revealed.png` after
the decoder finishes, followed immediately by the transcription. It should
not inspect `decoy.png`, count word lengths, construct candidate sentences,
or run a confirmation pass.

## What's inside

- `decoy-decode/SKILL.md` — self-contained: the instructions **and** the decoder
  (invert → heavy Gaussian low-pass → contrast/gamma stretch for the hidden
  layer, plus the high-pass remainder for the decoy layer). Claude writes the
  decoder to a file, runs it once, and reads both messages from the two images.
  No separate files needed.

## Notes

- The sandbox installs `opencv-python-headless` and `numpy` automatically.
- The skill runs once and produces exactly two files (`revealed.png` and
  `decoy.png`), but uses only `revealed.png` for transcription.
- The claude.ai workflow explicitly opens `revealed.png` once with the
  computer `view` tool. It forbids decoy comparison, word/letter counting,
  candidate-phrase generation, and repeated confirmation reads.
- Same decoder ships as a `/decoy-decode` command + skill for **Claude Code**
  and a `$decoy-decode` skill for **Codex**. See the repository root README.

## Rebuilding the zip

If you edit the skill, regenerate the archive with forward-slash paths:

```bash
cd claude-ai-skill && zip -r decoy-decode.zip decoy-decode
```
