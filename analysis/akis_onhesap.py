# Precomputes the motion field of every sequence.

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lagrange_kalicilik as LK
import akis_manifest as AM

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", DATA_DIR)
assert LK.FLOW in ("ilk", "tvl1"), LK.FLOW


def kos(split):
    src = np.load(os.path.join(DATA_DIR, f"meteonet_nw_{split}.npy"), mmap_mode="r")
    N = src.shape[0]
    dst_path = os.path.join(OUT_DIR, f"meteonet_nw_{split}_akis56_{LK.FLOW}.npy")
    tmp_path = dst_path + ".yaziliyor.npy"
    dst = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(N, 2, 128, 128))
    t0 = time.time()
    for i in range(N):
        x = np.asarray(src[i], dtype=np.float32)[::2]
        dst[i] = LK.flow(x[5], x[6])
        if i % 500 == 0:
            print(f"  {split} {i}/{N}  {time.time() - t0:.0f} s", flush=True)
    dst.flush(); del dst
    os.replace(tmp_path, dst_path)
    chk = np.load(dst_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.choice(N, size=min(16, N), replace=False):
        x = np.asarray(src[i], dtype=np.float32)[::2]
        assert np.array_equal(chk[i], LK.flow(x[5], x[6])), f"{split} dizi {i} uyusmuyor"
    print(f"{split}: {N} dizi, {time.time() - t0:.0f} s, denetim 16/16 bit-ayni -> {dst_path}", flush=True)
    m = AM.olustur(os.path.join(DATA_DIR, f"meteonet_nw_{split}.npy"), dst_path, LK.FLOW, "uretim",
                   uretim_denetimi="16/16 rastgele dizide akis yeniden hesaplandi, bit-ayni (akis_onhesap.py)")
    print(f"  manifest: akis {m['akis_sha256'][:16]}... -> {AM.manifest_yolu(dst_path)}", flush=True)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["test", "val", "train"]):
        kos(sp)
