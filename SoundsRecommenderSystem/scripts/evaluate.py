"""
Evaluate hybrid_recommender.tflite

Two modes:
  --mixes_file  : batch-evaluate all mixes from generated_mixes.txt (default)
  --mix         : evaluate a single mix

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --mixes_file data/generated_mixes.txt
    python scripts/evaluate.py --mix "ambience.river,ambience.rain,ambience.waterfall"
    python scripts/evaluate.py --tflite data/hybrid_recommender.tflite --N 5
"""

import argparse
import json
import numpy as np


def load_sounds_ids(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_mixes(path: str, sounds_ids: list) -> list[list[str]]:
    known = set(sounds_ids)
    mixes = []
    with open(path) as f:
        for line in f:
            sounds = [s.strip() for s in line.strip().split(",") if s.strip()]
            sounds = [s for s in sounds if s in known]
            if sounds:
                mixes.append(sounds)
    return mixes


def volume_vector(mix: list[str], sounds_ids: list, volume: float = 0.8) -> np.ndarray:
    """Build a [1, n_sounds] float32 L1-normalised volume vector from a mix."""
    v = np.zeros(len(sounds_ids), dtype=np.float32)
    sound_to_idx = {s: i for i, s in enumerate(sounds_ids)}
    for sid in mix:
        if sid in sound_to_idx:
            v[sound_to_idx[sid]] = volume
    total = v.sum()
    if total > 0:
        v /= total
    return v[np.newaxis, :]  # [1, n_sounds]


def load_interpreter(tflite_path: str):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def run_tflite(
    interpreter,
    input_vector: np.ndarray,
    N: int = 10,
    filter_already_liked: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TFLite inference and return top-N (indices, scores)."""
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    interpreter.set_tensor(inp_detail["index"], input_vector)
    interpreter.invoke()

    scores = interpreter.get_tensor(out_detail["index"]).squeeze()  # [n_sounds]

    if filter_already_liked:
        scores = scores.copy()
        scores[input_vector.squeeze() > 0] = -np.inf

    top_indices = np.argpartition(scores, -N)[-N:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return top_indices, scores[top_indices]


def print_recommendations(
    mix: list[str],
    indices: np.ndarray,
    scores: np.ndarray,
    sounds_ids: list,
) -> None:
    print(f"\nInput mix ({len(mix)} sounds): {', '.join(mix)}")
    print(f"  {'Rank':<5} {'Sound ID':<55} {'Score':>8}")
    print(f"  {'-'*4:<5} {'-'*54:<55} {'-'*8:>8}")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {rank:<5} {sounds_ids[int(idx)]:<55} {score:>8.4f}")


def evaluate_batch(
    interpreter,
    mixes: list[list[str]],
    sounds_ids: list,
    N: int,
    output_path: str,
) -> None:
    """Run all mixes and save results to a txt file in single-mix print format."""
    total = len(mixes)
    with open(output_path, "w") as f:
        for i, mix in enumerate(mixes, 1):
            inp = volume_vector(mix, sounds_ids)
            indices, scores = run_tflite(interpreter, inp, N=N)

            f.write(f"Input mix ({len(mix)} sounds): {', '.join(mix)}\n")
            f.write(f"  {'Rank':<5} {'Sound ID':<55} {'Score':>8}\n")
            f.write(f"  {'-'*4:<5} {'-'*54:<55} {'-'*8:>8}\n")
            for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                f.write(f"  {rank:<5} {sounds_ids[int(idx)]:<55} {score:>8.4f}\n")
            f.write("\n")

            if i % 500 == 0 or i == total:
                print(f"  {i:,}/{total:,} mixes evaluated...")

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid TFLite recommender")
    parser.add_argument("--tflite", default="data/hybrid_recommender.tflite")
    parser.add_argument("--sounds_ids", default="data/sounds_ids.json")
    parser.add_argument(
        "--mixes_file",
        default="data/generated_mixes.txt",
        help="Path to generated_mixes.txt for batch evaluation",
    )
    parser.add_argument(
        "--mix", default=None, help="Single mix (comma-separated) — overrides --mixes_file"
    )
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument(
        "--output", default="data/evaluation_results.txt", help="Output txt file for batch mode"
    )
    args = parser.parse_args()

    sounds_ids = load_sounds_ids(args.sounds_ids)
    interpreter = load_interpreter(args.tflite)

    # -- Single mix mode -----------------------------------------------------
    if args.mix:
        mix = [s.strip() for s in args.mix.split(",")]
        unknown = [s for s in mix if s not in sounds_ids]
        if unknown:
            print(f"Warning: ignoring unknown sounds: {unknown}")
        mix = [s for s in mix if s in sounds_ids]
        if not mix:
            raise ValueError("No valid sounds in mix.")
        indices, scores = run_tflite(interpreter, volume_vector(mix, sounds_ids), N=args.N)
        print_recommendations(mix, indices, scores, sounds_ids)
        return

    # -- Batch mode ----------------------------------------------------------
    print(f"Loading mixes from {args.mixes_file}...")
    mixes = load_mixes(args.mixes_file, sounds_ids)
    print(f"{len(mixes):,} mixes loaded")

    # Print first mix as a sanity check
    sample_inp = volume_vector(mixes[0], sounds_ids)
    sample_ids, sample_scores = run_tflite(interpreter, sample_inp, N=args.N)
    print_recommendations(mixes[0], sample_ids, sample_scores, sounds_ids)

    print(f"\nRunning batch evaluation ({len(mixes):,} mixes)...")
    evaluate_batch(interpreter, mixes, sounds_ids, N=args.N, output_path=args.output)


if __name__ == "__main__":
    main()
