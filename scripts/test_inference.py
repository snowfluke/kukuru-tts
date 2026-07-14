#!/usr/bin/env python3
"""
Kokoro Indonesian: Test Inference
==================================
Tests the fine-tuned Kokoro model with an Indonesian phonetic test set.

Usage:
    # Zero-config sanity check (downloads reference model + voicepack from HF)
    python scripts/test_inference.py

    # Convert checkpoint + run inference
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kukuru-tts/epoch_1st_00002.pth \
        --voicepack voices/awal_epoch3.pt \
        --output-dir test_output/epoch3

    # Use a previously converted model
    python scripts/test_inference.py \
        --model voices/kokoro_indonesian_epoch3.pth \
        --voicepack voices/awal_epoch3.pt

    # Run on CPU
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kukuru-tts/epoch_1st_00002.pth \
        --voicepack voices/awal_epoch3.pt \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

# Prefer the kokoro submodule over any pip-installed kokoro package
_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

# Default reference model used for zero-config verification runs.
# When neither --checkpoint/--model nor --voicepack is provided, the script
# lazily downloads these from HuggingFace into a local cache directory so a
# fresh clone can run `uv run scripts/test_inference.py` with no arguments.
# TODO: No Indonesian reference model is published yet — update these once
# the first fine-tuned Indonesian checkpoint lands on HuggingFace. Until
# then, pass --model/--checkpoint and --voicepack explicitly.
DEFAULT_REPO_ID = "snowfluke/kukuru-indonesian-reference"
DEFAULT_MODEL_FILENAME = "kukuru_indonesian.pth"
DEFAULT_VOICE_FILENAME = "voices/awal.pt"
MODEL_CACHE_DIR = "test_output/.model_cache"


def download_reference_file(filename: str, cache_dir: str = MODEL_CACHE_DIR) -> str:
    """Lazily download a reference file from the default HF repo into a cache.

    Returns the local path. If the file is already cached, no download occurs.
    """
    from huggingface_hub import hf_hub_download

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    resolved = cache / filename
    if resolved.exists():
        print(f"Using cached reference file: {resolved}")
    else:
        print(f"Downloading reference file from {DEFAULT_REPO_ID}: {filename}...")
        hf_hub_download(
            repo_id=DEFAULT_REPO_ID,
            filename=filename,
            local_dir=str(cache),
        )
    return str(resolved)

# Standard Indonesian phonetic test set — covers all major pronunciation challenges
TEST_SENTENCES = [
    # 1. ny (ɲ) and ng (ŋ)
    "Nyonya itu menyanyi dengan nyaring sambil menggenggam bunga.",
    # 2. c (tʃ) and j (dʒ) affricates
    "Cuaca cerah, jadi Joko jajan cendol di Cirebon.",
    # 3. Glottal stop (final k) and open vowel endings
    "Kakak dan bapak tidak masak enak malam ini.",
    # 4. e taling vs e pepet (e vs ə)
    "Enam ekor bebek berenang ke tepi telaga yang tenang.",
    # 5. Trilled/tapped r
    "Burung merpati terbang berputar-putar di udara segar.",
    # 6. sy (ʃ) and kh (x)
    "Masyarakat bersyukur setelah musyawarah akhir pekan.",
    # 7. Prosody: questions and exclamations
    "Mengapa kamu melakukan itu? Sungguh luar biasa!",
    # 8. Numbers
    "Harganya tepat seratus dua puluh tiga juta rupiah.",
]

# Phonemes that must appear somewhere in the test set output. Beyond core
# Indonesian coverage (ɲ, ŋ, ə), this fingerprints espeak-ng's conventions —
# ʧ/ʤ single-char affricates, ç for 'sy', ʔ from 'bebek' — so a phonemizer
# update that silently changes conventions (and would poison a dataset or
# break train/inference consistency) fails the check instead.
EXPECTED_PHONEMES = ["ɲ", "ŋ", "ə", "ʧ", "ʤ", "x", "ç", "ʔ"]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def check_frontend():
    """Verify that Indonesian espeak G2P produces sane phonemes for the test set.

    This operates entirely on phonemes and requires no model or voicepack.
    """
    try:
        from misaki import espeak
    except ImportError:
        print("ERROR: Could not import 'misaki.espeak'. Ensure misaki[en] and espeak-ng are installed.")
        sys.exit(1)

    print("\n=== Running Frontend G2P Checks ===")
    g2p = espeak.EspeakG2P(language="id")
    failures = 0
    all_phonemes = []

    for i, sentence in enumerate(TEST_SENTENCES):
        # EspeakG2P returns (phonemes, tokens)
        phonemes, _ = g2p(sentence)
        all_phonemes.append(phonemes)
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] Text: '{sentence}'")
        print(f"      Phonemes: '{phonemes}'")
        if not phonemes.strip():
            print("  ❌ FAIL: G2P returned empty phonemes.")
            failures += 1

    combined = "".join(all_phonemes)
    for ipa in EXPECTED_PHONEMES:
        if ipa in combined:
            print(f"  ✅ PASS: phoneme '{ipa}' present in test set output")
        else:
            print(f"  ❌ FAIL: expected Indonesian phoneme '{ipa}' not found in test set output.")
            failures += 1

    print("=" * 44)
    if failures == 0:
        print("🎉 ALL FRONTEND G2P CHECKS PASSED!\n")
        sys.exit(0)
    else:
        print(f"❌ {failures} FRONTEND G2P CHECKS FAILED.\n")
        sys.exit(1)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
):
    """Run inference on the Indonesian test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    print(f"  Config: {config_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with the Indonesian lang_code. Requires the kokoro
    # submodule fork to have id='id' in LANG_CODES (see docs).
    pipeline = KPipeline(lang_code="id", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate audio for each test sentence
    print(f"\nGenerating {len(TEST_SENTENCES)} phonetic test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Kokoro Indonesian model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint",
        help="Path to StyleTTS2 checkpoint (.pth) — will be converted automatically",
    )
    group.add_argument(
        "--model",
        help="Path to already-converted Kokoro-format weights (.pth). "
        f"If omitted (and no --checkpoint), downloads '{DEFAULT_MODEL_FILENAME}' "
        f"from {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--voicepack",
        required=False,
        help="Path to voicepack (.pt). If omitted, downloads "
        f"'{DEFAULT_VOICE_FILENAME}' from {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--config",
        default="training/config.json",
        help="Path to Kokoro config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory to save generated WAV files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--check-frontend",
        action="store_true",
        help="Only run fast G2P frontend verification checks, without generating audio",
    )

    args = parser.parse_args()

    if args.check_frontend:
        check_frontend()
        return

    # Resolve model path: convert checkpoint, use explicit model, or
    # fall back to the default reference model from HuggingFace.
    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_indonesian_converted.pth"),
        )
    elif args.model:
        model_path = args.model
    else:
        model_path = download_reference_file(DEFAULT_MODEL_FILENAME)

    # Resolve voicepack path: use explicit voicepack or fall back to the
    # default reference voicepack from HuggingFace.
    if args.voicepack:
        voicepack_path = args.voicepack
    else:
        voicepack_path = download_reference_file(DEFAULT_VOICE_FILENAME)

    run_inference(
        model_path=model_path,
        voicepack_path=voicepack_path,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
