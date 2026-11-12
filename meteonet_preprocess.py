# MeteoNet north-west radar files -> 128 x 128 sequences of 25 maps (training, validation, test).

import os
import glob
import numpy as np
import pandas as pd
from skimage.transform import resize as sk_resize

# settings

INPUT_ROOT = os.environ.get("INPUT_ROOT", "./meteonet_nw")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./out")
ZONE       = os.environ.get("ZONE", "NW")

SMOKE = os.environ.get("SMOKE", "0") == "1"

N_RAW_FRAMES = 25
STEP_MIN     = 5
IMG_SIZE     = 128
DOWNSAMPLE   = os.environ.get("DOWNSAMPLE", "bilinear").lower()
assert DOWNSAMPLE in ("bilinear", "blockmax"), \
    f"DOWNSAMPLE 'bilinear' | 'blockmax' olmali, gelen: {DOWNSAMPLE}"
STRIDE       = 25

MISSING_VALUE     = -1
MAX_MISSING_FRAC  = 0.20

RAW_TO_MMH   = 0.12
RAIN_MAX_MMH = float(os.environ.get("RAIN_MAX_MMH", "12.0"))
RAIN_LEVELS_MMH = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END", "2018-01-01"))
VAL_END   = pd.Timestamp(os.environ.get("VAL_END",   "2018-07-01"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# helpers

def find_npz(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    files = [f for f in files
             if "coord" not in os.path.basename(f).lower()
             and "quality" not in f.lower()]
    return files

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = z["data"]
    dates = pd.to_datetime(z["dates"])
    order = np.argsort(dates.values)
    return data[order], dates[order]

def preprocess_frame(raw):
    f = raw.astype(np.float32)
    f[f == MISSING_VALUE] = 0.0
    H, W = f.shape
    s = min(H, W)
    i0, j0 = (H - s) // 2, (W - s) // 2
    f = f[i0:i0 + s, j0:j0 + s]
    if DOWNSAMPLE == "blockmax":
        import torch
        import torch.nn.functional as F_t
        t = torch.from_numpy(f)[None, None]
        f = F_t.adaptive_max_pool2d(t, (IMG_SIZE, IMG_SIZE))[0, 0].numpy().astype(np.float32)
    else:
        f = sk_resize(f, (IMG_SIZE, IMG_SIZE), order=1,
                      preserve_range=True, anti_aliasing=True).astype(np.float32)
    f *= RAW_TO_MMH
    f = np.clip(f, 0.0, RAIN_MAX_MMH) / RAIN_MAX_MMH
    return f

def split_of(start_ts):
    if start_ts < TRAIN_END:  return "train"
    if start_ts < VAL_END:    return "val"
    return "test"

# main

def _iter_windows(files, materialize):
    buf_dates = []
    buf_frames = [] if materialize else None
    last_date = None
    expected = pd.Timedelta(minutes=STEP_MIN)
    n_emitted = 0

    for fi, path in enumerate(files):
        data, dates = load_npz(path)
        T = data.shape[0]
        for k in range(T):
            d = dates.iloc[k] if hasattr(dates, "iloc") else dates[k]
            raw = data[k]

            if (raw == MISSING_VALUE).mean() > MAX_MISSING_FRAC:
                buf_dates.clear()
                if materialize: buf_frames.clear()
                last_date = None
                continue
            if last_date is not None and (d - last_date) != expected:
                buf_dates.clear()
                if materialize: buf_frames.clear()
            buf_dates.append(d)
            if materialize:
                buf_frames.append(preprocess_frame(raw))
            last_date = d
            if len(buf_dates) == N_RAW_FRAMES:
                split = split_of(buf_dates[0])
                seq = np.stack(buf_frames, axis=0) if materialize else None
                yield split, seq
                n_emitted += 1
                del buf_dates[:STRIDE]
                if materialize: del buf_frames[:STRIDE]
                if SMOKE and n_emitted >= 5:
                    return
        if SMOKE and n_emitted >= 5:
            return
        if materialize:
            print(f"  [{fi+1}/{len(files)}] {os.path.basename(path)} yazıldı "
                  f"(toplam sequence: {n_emitted})")


def main():
    files = find_npz(INPUT_ROOT)
    assert files, f"INPUT_ROOT altında .npz bulunamadı: {INPUT_ROOT}"
    if SMOKE:
        files = files[:1]
    print(f"{len(files)} .npz dosyası işlenecek (SMOKE={SMOKE})")

    print("\n[1/2] Sayım geçişi (süreklilik/eksik mantığı, resize yok)...")
    counts = {"train": 0, "val": 0, "test": 0}
    for split, _ in _iter_windows(files, materialize=False):
        counts[split] += 1
    print(f"  sequence sayıları: {counts}")

    writers, idx, stats = {}, {}, {}
    for split, n in counts.items():
        if n == 0:
            print(f"  {split}: 0 sequence (boş, kaydedilmedi)")
            continue
        out = os.path.join(OUTPUT_DIR, f"meteonet_{ZONE.lower()}_{split}.npy")
        writers[split] = np.lib.format.open_memmap(
            out, mode="w+", dtype=np.float32,
            shape=(n, N_RAW_FRAMES, IMG_SIZE, IMG_SIZE))
        idx[split] = 0
        stats[split] = {"sum": 0.0, "n_elem": 0, "min": np.inf, "max": -np.inf, "gt0": 0}

    print("\n[2/2] Yazım geçişi (resize + streaming kayıt)...")
    for split, seq in _iter_windows(files, materialize=True):
        w = writers.get(split)
        if w is None:
            continue
        w[idx[split]] = seq
        idx[split] += 1
        s = stats[split]
        s["sum"] += float(seq.sum()); s["n_elem"] += seq.size
        s["min"] = min(s["min"], float(seq.min()))
        s["max"] = max(s["max"], float(seq.max()))
        s["gt0"] += int((seq > 0).sum())

    print("\n=== Çıktı ===")
    for split, w in writers.items():
        assert idx[split] == counts[split], \
            f"{split}: yazım ({idx[split]}) sayım ({counts[split]}) ile uyuşmadı"
        w.flush()
        out = os.path.join(OUTPUT_DIR, f"meteonet_{ZONE.lower()}_{split}.npy")
        s = stats[split]
        mean = s["sum"] / s["n_elem"] if s["n_elem"] else 0.0
        print(f"  {split}: {w.shape}  -> {out}")
        print(f"     min={s['min']:.4f} max={s['max']:.4f} mean={mean:.5f} "
              f"(>0 oran: {s['gt0'] / s['n_elem']:.4f})")
        del w

    csi_norm = [round(l / RAIN_MAX_MMH, 6) for l in RAIN_LEVELS_MMH]
    print("\n=== 15a/15b'ye YAPIŞTIR (normalize [0,1] eşikler) ===")
    print(f"  # RAIN_MAX_MMH={RAIN_MAX_MMH}, seviyeler(mm/h)={RAIN_LEVELS_MMH}")
    print(f"  CSI_THRESHOLDS = {csi_norm}")
    print(f"  # B-MSE bin sınırları da bu 6 eşiğe göre yeniden tanımlanmalı")
    print(f"  # (bmse_loss içindeki 16/255, 74/255 ... yerine yukarıdaki değerler).")
    print("\nTamamlandı.")

if __name__ == "__main__":
    main()
