"""
Convert raw audio into cached log-mel spectrograms.

Usage (run from the repo root):
    python -m src.data.preprocess --dataset fma --limit 200    # smoke test (writes to *_smoke folder)
    python -m src.data.preprocess --dataset fma                # full fma_small
    python -m src.data.preprocess --dataset gtzan              # Hip-Hop / Pop / Rock only (run AFTER fma)

Expected layout:
    data/raw_data/fma_small/000/000002.mp3 ...
    data/raw_data/fma_metadata/tracks.csv
    data/raw_data/gtzan/.../hiphop/hiphop.00000.wav ...

Outputs (in data/spectrograms/):
    {train,val,test}_x.npy    (N, 128, 1292) float16, un-normalized dB
    {train,val,test}_y.npy    (N,) int64, index into genres.json
    {train,val,test}_ids.npy  (N,) int64, FMA track IDs
    stats.npz                 per-mel-bin mean/std from the TRAIN split only
    dropped.csv               files that failed to decode, with the reason
    gtzan_{x,y,ids}.npy       same format, for the cross-dataset test

The script is resumable: work is saved in shards of 500 clips, and finished shards
are skipped if you rerun the same command after a crash or disconnect.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# ---- Preprocessing parameters (every change invalidates cached results) ----
SR = 22050
DURATION = 30
N_SAMPLES = SR * DURATION            # 661,500
N_FFT = 2048
HOP = 512
N_MELS = 128
N_FRAMES = 1 + N_SAMPLES // HOP      # 1292 (librosa centers frames)
SHARD_SIZE = 500

PARAMS = dict(sr=SR, duration=DURATION, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
              power=2.0, ref=1.0, amin=1e-10, top_db=None, shard_size=SHARD_SIZE,
              librosa=librosa.__version__)

FMA_SPLITS = {"training": "train", "validation": "val", "test": "test"}
GTZAN_FOLDERS = {"hiphop": "Hip-Hop", "pop": "Pop", "rock": "Rock"}  # overlap with FMA


# --------------------------------------------------------------------------
# Core: one audio file -> one log-mel spectrogram
# --------------------------------------------------------------------------
def audio_to_logmel(path):
    """Return (array, None) on success or (None, reason) on failure.

    array is (128, 1292) float16 in dB. A FIXED reference (ref=1.0) is used rather than
    ref=np.max, so loudness information is preserved and gain augmentation is meaningful.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SR, mono=True, duration=DURATION)
        if y.size < SR:  # under 1 second: treat as broken
            return None, f"too short ({y.size} samples)"
        y = librosa.util.fix_length(y, size=N_SAMPLES)  # pad with zeros or trim to 30 s
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
        db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        if db.shape != (N_MELS, N_FRAMES) or not np.isfinite(db).all():
            return None, f"bad output (shape {db.shape})"
        return db.astype(np.float16), None
    except Exception as e:  # corrupt mp3, missing file, decoder error...
        return None, f"{type(e).__name__}: {e}"[:200]


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------
def load_fma_table(raw_dir):
    """Rows for fma_small only, with columns: split, genre. Indexed by track_id."""
    tracks = pd.read_csv(raw_dir / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"]
    table = pd.DataFrame({"split": small[("set", "split")],
                          "genre": small[("track", "genre_top")]}).dropna()
    table.index.name = "track_id"
    return table.sort_index()


def fma_path(audio_dir, track_id):
    s = f"{track_id:06d}"                      # 2 -> 000002, folder is the first 3 digits
    return audio_dir / s[:3] / f"{s}.mp3"


def load_genres(path, found=None):
    """genres.json is a list; a genre's index is its position. Created from FMA if empty."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        genres = json.loads(p.read_text())
    elif found is not None:
        genres = sorted(found)
        p.write_text(json.dumps(genres, indent=2))
        print(f"Wrote {p} -> {genres}")
    else:
        raise SystemExit(f"{p} is missing or empty. Run '--dataset fma' first.")
    return {g: i for i, g in enumerate(genres)}


def check_params(out_dir):
    """Refuse to mix shards made with different settings."""
    p = out_dir / "preprocess_params.json"
    if p.exists() and json.loads(p.read_text()) != json.loads(json.dumps(PARAMS)):
        raise SystemExit(f"{p} differs from the current settings. Delete {out_dir} and rerun.")
    p.write_text(json.dumps(PARAMS, indent=2))


# --------------------------------------------------------------------------
# Sharded, resumable processing
# --------------------------------------------------------------------------
def process_items(items, shard_dir, prefix, n_jobs):
    """items: list of (id, path, label). Writes one .npz per SHARD_SIZE items."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), SHARD_SIZE):
        shard_path = shard_dir / f"{prefix}_{start // SHARD_SIZE:04d}.npz"
        if shard_path.exists():
            continue  # finished in an earlier run
        chunk = items[start:start + SHARD_SIZE]
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(audio_to_logmel)(path) for _, path, _ in chunk)

        xs, ids, ys, bad_ids, bad_why = [], [], [], [], []
        for (tid, _, label), (arr, why) in zip(
                chunk, tqdm(results, total=len(chunk), desc=f"{prefix} shard {start // SHARD_SIZE}")):
            if arr is None:
                bad_ids.append(tid); bad_why.append(why)
            else:
                xs.append(arr); ids.append(tid); ys.append(label)

        x = np.stack(xs) if xs else np.empty((0, N_MELS, N_FRAMES), np.float16)
        tmp = shard_path.with_suffix(".tmp.npz")  # write then rename, so a crash never leaves a half shard
        np.savez(tmp, x=x, ids=np.array(ids, dtype=np.int64), y=np.array(ys, dtype=np.int64),
                 failed_ids=np.array(bad_ids, dtype=np.int64), failed_reasons=np.array(bad_why, dtype=str))
        os.replace(tmp, shard_path)


def merge_shards(shard_dir, prefix, out_dir, name):
    """Combine shards into {name}_x/y/ids.npy. Returns list of (id, reason) for dropped files."""
    paths = sorted(shard_dir.glob(f"{prefix}_*.npz"))
    ids, ys, failed = [], [], []
    for p in paths:
        with np.load(p) as z:
            ids.append(z["ids"]); ys.append(z["y"])
            failed += list(zip(z["failed_ids"].tolist(), z["failed_reasons"].tolist()))
    ids = np.concatenate(ids) if ids else np.array([], dtype=np.int64)
    ys = np.concatenate(ys) if ys else np.array([], dtype=np.int64)

    # Memory-mapped output, so the full array never has to sit in RAM twice
    x = np.lib.format.open_memmap(out_dir / f"{name}_x.npy", mode="w+",
                                  dtype=np.float16, shape=(len(ids), N_MELS, N_FRAMES))
    i = 0
    for p in paths:
        with np.load(p) as z:
            xb = z["x"]
        x[i:i + len(xb)] = xb
        i += len(xb)
    x.flush()
    del x
    np.save(out_dir / f"{name}_y.npy", ys)
    np.save(out_dir / f"{name}_ids.npy", ids)
    return failed


def compute_stats(x_path):
    """Per-mel-bin mean/std over all frames of the TRAIN split (computed in chunks)."""
    x = np.load(x_path, mmap_mode="r")
    total = np.zeros(N_MELS, np.float64)
    total_sq = np.zeros(N_MELS, np.float64)
    count = 0
    for i in range(0, len(x), 128):
        b = np.asarray(x[i:i + 128], dtype=np.float64)   # (b, 128, T)
        total += b.sum(axis=(0, 2))
        total_sq += (b ** 2).sum(axis=(0, 2))
        count += b.shape[0] * b.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def report(out_dir, name, genre_names):
    y = np.load(out_dir / f"{name}_y.npy")
    counts = np.bincount(y, minlength=len(genre_names))
    print(f"  {name:6s} N={len(y):5d}  " + "  ".join(f"{g}={c}" for g, c in zip(genre_names, counts)))


# --------------------------------------------------------------------------
# Dataset-level drivers
# --------------------------------------------------------------------------
def run_fma(args, out_dir):
    raw = Path(args.raw_dir)
    table = load_fma_table(raw)
    label_map = load_genres(args.genres_file, table["genre"].unique())
    genre_names = list(label_map)
    audio_dir = raw / "fma_small"
    dropped = []

    for fma_split, name in FMA_SPLITS.items():
        rows = table[table["split"] == fma_split]
        if args.limit:
            rows = rows.sample(n=min(args.limit, len(rows)), random_state=0).sort_index()
        items = [(tid, fma_path(audio_dir, tid), label_map[g]) for tid, g in rows["genre"].items()]
        print(f"\n[{name}] {len(items)} clips")
        process_items(items, out_dir / "shards", name, args.n_jobs)
        dropped += [(name, tid, why) for tid, why in merge_shards(out_dir / "shards", name, out_dir, name)]

    pd.DataFrame(dropped, columns=["split", "track_id", "reason"]).to_csv(out_dir / "dropped.csv", index=False)
    mean, std = compute_stats(out_dir / "train_x.npy")
    np.savez(out_dir / "stats.npz", mean=mean, std=std)

    print(f"\nDone. Dropped {len(dropped)} clips (see dropped.csv). Final counts:")
    for name in FMA_SPLITS.values():
        report(out_dir, name, genre_names)


def run_gtzan(args, out_dir):
    label_map = load_genres(args.genres_file)
    genre_names = list(label_map)
    missing = [g for g in GTZAN_FOLDERS.values() if g not in label_map]
    if missing:
        raise SystemExit(f"{missing} not found in genres.json {genre_names}")
    gtzan_dir = Path(args.gtzan_dir or Path(args.raw_dir) / "gtzan")
    items, files = [], []
    for folder, genre in GTZAN_FOLDERS.items():
        for p in sorted(gtzan_dir.rglob(f"{folder}/*.wav")):
            items.append((len(items), p, label_map[genre]))
            files.append((len(items) - 1, str(p)))
    if not items:
        raise SystemExit(f"No GTZAN .wav files found under {gtzan_dir}")
    if args.limit:
        items = items[:args.limit]
    print(f"\n[gtzan] {len(items)} clips")
    process_items(items, out_dir / "shards", "gtzan", args.n_jobs)
    failed = merge_shards(out_dir / "shards", "gtzan", out_dir, "gtzan")
    pd.DataFrame(files, columns=["id", "path"]).to_csv(out_dir / "gtzan_files.csv", index=False)
    pd.DataFrame(failed, columns=["track_id", "reason"]).to_csv(out_dir / "gtzan_dropped.csv", index=False)
    print(f"Done. Dropped {len(failed)} clips.")
    report(out_dir, "gtzan", genre_names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["fma", "gtzan"], required=True)
    ap.add_argument("--raw_dir", default="data/raw_data")
    ap.add_argument("--out_dir", default="data/spectrograms")
    ap.add_argument("--gtzan_dir", default=None, help="folder containing hiphop/, pop/, rock/ (searched recursively)")
    ap.add_argument("--genres_file", default="genres.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N clips per split")
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.limit:  # keep smoke-test shards away from the real cache
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
        print(f"--limit set: writing to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_params(out_dir)

    (run_fma if args.dataset == "fma" else run_gtzan)(args, out_dir)


if __name__ == "__main__":
    main()