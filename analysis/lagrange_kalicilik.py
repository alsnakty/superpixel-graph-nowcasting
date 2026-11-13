# Motion field from optical flow and backward advection with one displacement vector per pixel (Lagrangian persistence).

import sys, json, time
import numpy as np
from scipy.ndimage import map_coordinates
import os
from skimage.registration import optical_flow_ilk, optical_flow_tvl1

FLOW = os.environ.get("FLOW", "ilk")

SPLIT = os.environ.get("SPLIT", "test")
SETS = {
    "meteonet": (f"./data/meteonet_nw_{SPLIT}.npy", True,
                 [0.1/12, 0.5/12, 1.0/12, 2.0/12, 5.0/12, 10.0/12]),
    "sevir":    (f"./data/sevir_vil_{SPLIT}_subset.npy", False,
                 [16/255, 74/255, 133/255, 160/255, 181/255, 219/255]),
}
H = W = 128
RR, CC = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")


def flow(prev, last):
    if FLOW == "tvl1":
        return optical_flow_tvl1(last, prev)
    return optical_flow_ilk(last, prev)


def advect(img, v, k):
    coords = np.stack([RR + k * v[0], CC + k * v[1]])
    return map_coordinates(img, coords, order=1, mode="constant", cval=0.0)


def pool(a, p):
    n = a.shape[0]
    return a.reshape(n, H // p, p, W // p, p).max(axis=(2, 4))


class Sayac:
    def __init__(self, T):
        self.c = {g: np.zeros((T, 6, 3)) for g in ("piksel", "pool4", "pool16")}

    def ekle(self, pred, obs, thr, k):
        for g, p in (("piksel", 1), ("pool4", 4), ("pool16", 16)):
            pp, oo = (pred, obs) if p == 1 else (pool(pred, p), pool(obs, p))
            for j, t in enumerate(thr):
                a = pp >= t; b = oo >= t
                self.c[g][j, k] += [(a & b).sum(), (a & ~b).sum(), (~a & b).sum()]

    def ozet(self):
        out = {}
        for g, c in self.c.items():
            tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
            agg = tp.sum(1) / np.maximum(tp.sum(1) + fp.sum(1) + fn.sum(1), 1)
            lead = (tp / np.maximum(tp + fp + fn, 1)).mean(0)
            T, F, M = tp.sum(1), fp.sum(1), fn.sum(1)
            out[g] = {"csi_ort6": float(agg.mean()), "csi_esik": agg.tolist(), "csi_adim": lead.tolist(),
                      "bias_esik": ((T + F) / np.maximum(T + M, 1)).tolist(),
                      "pod_esik": (T / np.maximum(T + M, 1)).tolist(),
                      "far_esik": (F / np.maximum(T + F, 1)).tolist()}
        return out


def kos(name, path, stride2, thr, n_seq=None, zero_flow=False):
    arr = np.load(path, mmap_mode="r")
    N = arr.shape[0] if not n_seq else min(n_seq, arr.shape[0])
    E, L = Sayac(len(thr)), Sayac(len(thr))
    hiz = []
    t0 = time.time()
    for s in range(0, N, 32):
        x = np.asarray(arr[s:min(s + 32, N)], dtype=np.float32)
        if stride2:
            x = x[:, ::2]
        assert x.shape[1] == 13
        last = x[:, 6]
        vs = []
        for i in range(x.shape[0]):
            v = np.zeros((2, H, W), np.float32) if zero_flow else flow(x[i, 5], x[i, 6])
            vs.append(v)
            if (last[i] >= thr[0]).mean() >= 0.01:
                hiz.append(float(np.hypot(v[0], v[1])[last[i] >= thr[0]].mean()))
        for k in range(6):
            obs = x[:, 7 + k]
            lag = np.stack([advect(last[i], vs[i], k + 1) for i in range(x.shape[0])])
            E.ekle(last, obs, thr, k)
            L.ekle(lag, obs, thr, k)
    e, l = E.ozet(), L.ozet()
    G = {g: float(np.mean([l[g]["csi_adim"][k] - e[g]["csi_adim"][k] for k in (2, 3, 4, 5)]))
         for g in e}
    return {"N_seq": N, "sure_s": round(time.time() - t0, 1), "euler": e, "lagrange": l, "G_t30_60": G,
            "yagisli_alanda_ort_hiz_px_adim": float(np.mean(hiz)) if hiz else None,
            "hiz_olculen_dizi": len(hiz)}


def sentetik_test():
    yy, xx = RR.astype(np.float32), CC.astype(np.float32)
    def leke(r, c):
        return np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 6.0 ** 2)).astype(np.float32)
    seq = np.stack([leke(40 + 2 * t, 30 + 3 * t) for t in range(13)])
    v = flow(seq[5], seq[6])
    m = seq[6] > 0.3
    vr, vc = float(v[0][m].mean()), float(v[1][m].mean())
    fc = advect(seq[6], v, 6)
    err_lag = float(np.abs(fc - seq[12]).mean()); err_eul = float(np.abs(seq[6] - seq[12]).mean())
    r = {"akis_satir": vr, "akis_sutun": vc, "beklenen": [-2, -3],
         "t+60_MAE_lagrange": err_lag, "t+60_MAE_euler": err_eul}
    assert abs(vr + 2) < 0.2 and abs(vc + 3) < 0.2, r
    assert err_lag < 0.2 * err_eul, ('tasima gozlemi yakalamiyor', r)
    return r


if __name__ == "__main__":
    out_path = sys.argv[1]
    n_seq = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res = {"sentetik": sentetik_test()}
    print("sentetik:", res["sentetik"], flush=True)
    z = kos("meteonet", *SETS["meteonet"], n_seq=n_seq, zero_flow=True)
    res["dogrulama_sifir_akis_meteonet"] = {"piksel": z["euler"]["piksel"]["csi_ort6"],
                                            "pool16": z["euler"]["pool16"]["csi_ort6"],
                                            "lagrange_piksel_ayni_mi": z["lagrange"]["piksel"]["csi_ort6"]}
    print("sifir akis:", res["dogrulama_sifir_akis_meteonet"], flush=True)
    for name, (p, st, thr) in SETS.items():
        res[name] = kos(name, p, st, thr, n_seq=n_seq)
        print(name, "G:", res[name]["G_t30_60"], "sure", res[name]["sure_s"], flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
