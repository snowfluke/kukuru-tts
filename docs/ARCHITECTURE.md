# Architecture and Compatibility Notes

Technical reference for Kokoro-82M fine-tuning compatibility.

For how-to training steps, use `TRAINING_GUIDE.md`.

## Kokoro-82M Component Layout

Reference component sizes used for checkpoint compatibility checks:

| Component | Parameters |
|---|---|
| bert (PLBERT) | 6.29M |
| bert_encoder | 0.39M |
| predictor | 16.19M |
| text_encoder | 5.61M |
| decoder (ISTFTNet) | 53.28M |
| Total | 81.76M |

Voicepack target shape:
- `[510, 1, 256]` (float32)

## weight_norm API Compatibility

### Why it matters

Old API (`torch.nn.utils.weight_norm`) and new API (`torch.nn.utils.parametrizations.weight_norm`) create different state-dict key layouts.

If StyleTTS2 is trained with old API and inference expects new API, checkpoint loading can be brittle and may fail silently under non-strict loading paths.

### Required status

StyleTTS2 patched files must use new parametrizations API:
- `StyleTTS2/models.py`
- `StyleTTS2/Modules/istftnet.py`
- `StyleTTS2/Modules/hifigan.py`
- `StyleTTS2/Modules/discriminators.py`

## Symbol Mapping Compatibility

Kokoro and default StyleTTS2 use different token index assignments.

Implication:
- same symbol set size does not imply index compatibility

Requirement:
- `StyleTTS2/text_utils.py` must use Kokoro mapping (`kokoro_symbols.py`)

## Indonesian G2P Notes

- G2P backend: `misaki` + `espeak-ng`
- Indonesian code path uses `espeak.EspeakG2P(language='id')`
- Inference side: the kokoro fork must add `id='id'` to `LANG_CODES`
  (`kokoro/kokoro/pipeline.py`) — one line; the generic espeak branch handles
  the rest. Training and inference must use the same G2P.

## Indonesian Phoneme Compatibility

The symbol set below is what `EspeakG2P(language='id')` actually emits
(measured with espeak-ng 1.52 via `espeakng-loader`, misaki 0.9.4), with IDs
verified against `hexgrad/Kokoro-82M` `config.json`. Every emitted symbol is
in Kokoro's 178-token vocab.

| Sound | Emitted IPA | Unicode | Kokoro ID |
|-------|-------------|---------|-----------|
| ny | `ɲ` | U+0272 | 114 |
| ng | `ŋ` | U+014B | 112 |
| c affricate | `ʧ` (single char) | U+02A7 | 133 |
| j affricate | `ʤ` (single char) | U+02A4 | 82 |
| sy | `ç` (espeak quirk, not `ʃ`) | U+00E7 | 78 |
| kh | `x` | U+0078 | 66 |
| e pepet (schwa) | `ə` | U+0259 | 83 |
| e taling | `ɛ` (espeak prefers open-mid) | U+025B | 86 |
| o | `o` / `ɔ` | U+006F / U+0254 | 57 / 76 |
| r | `r` (plain, not normalized to `ɾ`) | U+0072 | 60 |
| glottal stop (final k) | `ʔ` (inconsistent, see below) | U+0294 | 148 |
| ai diphthong | `I` (misaki notation for aɪ) | U+0049 | 25 |
| au diphthong | `W` (misaki notation for aʊ) | U+0057 | 39 |
| w | `w` | U+0077 | 65 |
| y (approximant) | `j` | U+006A | 52 |
| stress | `ˈ` / `ˌ` | U+02C8 / U+02CC | 156 / 157 |

### Missing symbols

None — verified empirically by phonemizing the full test-sentence set plus a
loanword corpus (f/v/z/sy/kh words, numbers, dates) and diffing the emitted
character set against the vocab. `PHONEME_FIXUPS` in
`scripts/prepare_dataset.py` therefore stays empty; if a future
espeak-ng/misaki bump emits something new, `scripts/prepare_training.py
verify` reports unknown symbols and `scripts/test_inference.py
--check-frontend` fingerprints the conventions.

## Measured espeak-ng behavior (production notes)

espeak-ng's Indonesian rules are usable but heuristic where the language is
lexical. Measured on a ground-truth word list (KBBI pronunciations):

- **e pepet vs e taling is the main quality risk.** espeak applies a
  positional rule — stressed `e` → `ɛ`, unstressed/final `e` → `ə` — but the
  real distinction is lexical. Measured: only 5/20 common pepet words correct
  (`teman`→`tˈɛman`, `enam`→`ˈɛnam`, `empat`→`ˈɛmpat`, `besar`→`bˈɛsar` are
  all wrong), 12/16 taling words correct (word-final taling collapses to
  schwa: `sate`→`sˈatə`, `kue`→`kˈuə`). Consequence: the `ɛ` and `ə` tokens
  each cover a mix of true [ɛ] and [ə] audio, which muddies vowel quality for
  exactly these high-frequency words. No open espeak-ng issue tracks this, so
  don't expect an upstream fix.
- **Glottal stop is inconsistent:** `bebek`→`bˈɛbɛʔ` but `tidak`→`tˈidak`,
  `kakak`→`kˈakak`. Tolerable as long as it is the *same* G2P at training and
  inference — the model learns espeak's convention, not the dictionary's.
- **Numbers and dates are read correctly in Indonesian**
  (`123`→`sərˈatus dˈuapˌuluhtˈiɡa`).
- **Stress marks are emitted** and are in Kokoro's vocab; do not strip them.

### Alternative G2P (quality upgrade path)

[Wikidepia/g2p-id](https://github.com/Wikidepia/g2p-id) (MIT, maintained)
fixes exactly the two weak spots above lexically: a `schwa_dict.csv` lexicon
+ CRF syllabifier for the e distinction, and a rule for final-k glottal
stops. Adopting it is a project, not a swap: its output must be mapped to
Kokoro vocab symbols (it emits no stress marks, keeps some orthographic
symbols) and the *same* G2P must be implemented at inference time in the
kokoro fork — the German ancestor of this repo solved the analogous problem
with a misaki fork carrying pronunciation overrides. Never mix G2Ps between
dataset prep and runtime. A lighter middle path: keep espeak and patch only
the mislabeled `e` words using g2p-id's `schwa_dict.csv` as an override
lexicon, applied identically in `prepare_dataset.py` and the fork's G2P.
[kirralabs/indonesian-fonem](https://github.com/kirralabs/indonesian-fonem)
is a useful reference table of the Indonesian phoneme inventory.

### Diacritics (stress markers)

| Symbol | Meaning | Kokoro ID |
|--------|---------|-----------|
| `ˈ` | primary stress | 156 |
| `ˌ` | secondary stress | 157 |

These are produced by `espeak-ng` and are in Kokoro's vocabulary. Do not strip them.

## Sequence Length Constraint

- PLBERT max position embeddings: 512
- Practical training cap: 510 cleaned tokens

Samples above this should be filtered before batching.

## Inference Packaging Notes

When exporting trained checkpoints for `KModel`, ensure the expected components are present and keys align with Kokoro inference code:
- `bert`
- `bert_encoder`
- `predictor`
- `text_encoder`
- `decoder`

Use `scripts/test_inference.py` to verify conversion and produce sample outputs.
