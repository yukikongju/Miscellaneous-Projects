"""
Generate ranked recommendation lists using hybrid_recommender.tflite.

Two modes:
  --mixes_file  : batch-generate for all mixes from generated_mixes.txt (default)
  --mix         : generate for a single mix

Outputs both:
  - CSV  (--output_csv): input_mix, rank, sound_id, score
  - Text (--output_txt): human-readable ranked table

Usage:
    python scripts/hybrid_recommender_generation.py
    python scripts/hybrid_recommender_generation.py --mixes_file data/generated_mixes.txt
    python scripts/hybrid_recommender_generation.py --mix "ambience.river,ambience.rain" --N 5
    python scripts/hybrid_recommender_generation.py --tflite data/hybrid_recommender.tflite --N 10
"""

import argparse
import csv
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


def write_csv(rows: list[dict], path: str, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_mix", "rank", "sound_id", "score"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_text(
    mix: list[str], indices: np.ndarray, scores: np.ndarray, sounds_ids: list, f
) -> None:
    f.write(f"Input mix ({len(mix)} sounds): {', '.join(mix)}\n")
    f.write(f"  {'Rank':<5} {'Sound ID':<55} {'Score':>8}\n")
    f.write(f"  {'-'*4:<5} {'-'*54:<55} {'-'*8:>8}\n")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        f.write(f"  {rank:<5} {sounds_ids[int(idx)]:<55} {score:>8.4f}\n")
    f.write("\n")


def generate(
    interpreter,
    mixes: list[list[str]],
    sounds_ids: list,
    N: int,
    csv_path: str,
    txt_path: str,
) -> None:
    total = len(mixes)
    pending_rows: list[dict] = []
    first_flush = True

    with open(txt_path, "w") as txt_f:
        for i, mix in enumerate(mixes, 1):
            inp = volume_vector(mix, sounds_ids)
            indices, scores = run_tflite(interpreter, inp, N=N)

            mix_key = "|".join(sorted(mix))
            for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                pending_rows.append(
                    {
                        "input_mix": mix_key,
                        "rank": rank,
                        "sound_id": sounds_ids[int(idx)],
                        "score": round(float(score), 6),
                    }
                )

            write_text(mix, indices, scores, sounds_ids, txt_f)

            if i % 500 == 0 or i == total:
                write_csv(pending_rows, csv_path, write_header=first_flush)
                first_flush = False
                pending_rows = []
                print(f"  {i:,}/{total:,} mixes processed...")

    print(f"\nCSV  → {csv_path}")
    print(f"Text → {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ranked lists from hybrid TFLite recommender"
    )
    parser.add_argument("--tflite", default="data/hybrid_recommender.tflite")
    parser.add_argument("--sounds_ids", default="data/sounds_ids.json")
    parser.add_argument("--mixes_file", default="data/generated_mixes.txt")
    parser.add_argument(
        "--mix", default=None, help="Single mix (comma-separated) — overrides --mixes_file"
    )
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--output_csv", default="data/ranked_recommendations.csv")
    parser.add_argument("--output_txt", default="data/ranked_recommendations.txt")
    args = parser.parse_args()

    sounds_ids = load_sounds_ids(args.sounds_ids)
    interpreter = load_interpreter(args.tflite)

    # -- Single mix mode -------------------------------------------------------
    if args.mix:
        mix = [s.strip() for s in args.mix.split(",")]
        unknown = [s for s in mix if s not in sounds_ids]
        if unknown:
            print(f"Warning: ignoring unknown sounds: {unknown}")
        mix = [s for s in mix if s in sounds_ids]
        if not mix:
            raise ValueError("No valid sounds in mix.")

        inp = volume_vector(mix, sounds_ids)
        print(inp.shape)
        indices, scores = run_tflite(interpreter, inp, N=args.N)

        print(f"\nInput mix ({len(mix)} sounds): {', '.join(mix)}")
        print(f"  {'Rank':<5} {'Sound ID':<55} {'Score':>8}")
        print(f"  {'-'*4:<5} {'-'*54:<55} {'-'*8:>8}")
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            print(f"  {rank:<5} {sounds_ids[int(idx)]:<55} {score:>8.4f}")

        mix_key = "|".join(sorted(mix))
        rows = [
            {
                "input_mix": mix_key,
                "rank": r,
                "sound_id": sounds_ids[int(i)],
                "score": round(float(s), 6),
            }
            for r, (i, s) in enumerate(zip(indices, scores), 1)
        ]
        write_csv(rows, args.output_csv, write_header=True)
        print(f"\nCSV  → {args.output_csv}")
        return

    # -- Batch mode ------------------------------------------------------------
    print(f"Loading mixes from {args.mixes_file}...")
    mixes = load_mixes(args.mixes_file, sounds_ids)
    print(f"{len(mixes):,} mixes loaded")
    print(f"Generating top-{args.N} recommendations...\n")

    generate(
        interpreter, mixes, sounds_ids, N=args.N, csv_path=args.output_csv, txt_path=args.output_txt
    )


if __name__ == "__main__":
    main()
