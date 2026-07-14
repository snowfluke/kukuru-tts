# kukuru-tts

<img src="docs/images/kikiri-tts-logo.png" alt="logo" width="150">

> [!NOTE]
> This repository is a fork of [kikiri-tts](https://github.com/semidark/kikiri-tts)
> (formerly `kokoro-deutsch`), retargeted from German to Indonesian.

Training recipe for fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) for Indonesian with a patched [StyleTTS2](https://github.com/yl4579/StyleTTS2) submodule.

## What This Is

- A reproducible fine-tuning workflow (dataset prep -> Stage 1 -> Stage 2 -> voicepack extraction)
- Original scripts for data preparation and checkpoint/voicepack conversion
- A patched `StyleTTS2/` submodule with the fixes required for stable Stage 2 training

## What This Is Not

- Not a general-purpose Kokoro replacement repository
- Not a bundled upstream mirror of `demo/`, `examples/`, `kokoro.js/`, or `tests/`
- Not a redistributable training dataset

## Start Here

### I want to train my own Indonesian voice

Start with `docs/TRAINING_GUIDE.md`.

### I am debugging training failures

Go to `docs/TROUBLESHOOTING.md`.

### I want architecture details and compatibility notes

See `docs/ARCHITECTURE.md`.

## Status

The pipeline structure is inherited working from the German recipe:

`Dataset preparation -> Weight conversion -> Stage 1 -> Stage 2 -> Voicepack extraction -> KModel inference`

No Indonesian checkpoint has been trained or published yet — this fork is the
training recipe for producing one. The German models the original recipe
produced live at the [kikiri-tts](https://huggingface.co/kikiri-tts)
HuggingFace organization.

## Published Models & Voices

*None yet. Indonesian base model and voices will be listed here once trained.*

## Running Verification Tests

To run a fast text-to-speech frontend sanity check — e.g. after updating
phonemizer packages like `misaki`, bumping dependencies, or making model
changes — run the G2P frontend checks (no model download required):

```bash
uv run scripts/test_inference.py --check-frontend
```

This phonemizes the standard Indonesian phonetic test sentences with
espeak-ng and verifies core Indonesian phonemes (ɲ, ŋ, ə, ʧ, ʤ, x, ç, ʔ)
appear in the output — fingerprinting the phonemizer's conventions so a
silent G2P change fails loudly. See "Measured espeak-ng behavior" in
`docs/ARCHITECTURE.md` for what espeak gets right and wrong for Indonesian.

To run full inference against a trained checkpoint or voice, pass explicit
paths (the zero-config download mode is disabled until the first Indonesian
reference model is published):

```bash
uv run scripts/test_inference.py \
    --model voices/kokoro_indonesian_epoch3.pth \
    --voicepack voices/awal_epoch3.pt \
    --device cpu
```

It runs on CPU automatically when no GPU is available, so it works on any
machine and in CI/CD pipelines. Audio is written to `test_output/`.

## Repository Layout

```text
kokoro/              # Kokoro fork submodule (contains the `kokoro/` Python package)
StyleTTS2/           # Patched training code (git submodule: semidark/StyleTTS2)
scripts/             # Dataset prep, voicepack extraction, inference testing
configs/             # Training config(s)
docs/                # Training guide, troubleshooting, architecture notes
training/            # Local training artifacts metadata (audio excluded)
```

## Contributing

Contributions are welcome, especially:

- Reproducible runs on public Indonesian datasets
- Fine-tuning recipes for other languages
- Training stability and quality improvements

## Attribution

See `NOTICE` for upstream attribution and license details.

## License

Apache License 2.0 — see `LICENSE`.
