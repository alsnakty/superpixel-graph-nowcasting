# Scores a forecast on the validation or test set (ConvLSTM, Lagrangian and Eulerian persistence).

import os, sys, json, time, importlib.util
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATA_DIR", os.path.join(ROOT, "data"))
os.environ.setdefault("FLOW_DIR", os.environ["DATA_DIR"])
os.environ.setdefault("LK_DIR", os.path.join(ROOT, "analysis"))
_tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "ortak_degerlendirme_out")
os.environ.setdefault("OUT_DIR", _tmp)
os.environ.setdefault("FIG_DIR", _tmp)
os.environ["SELF_TEST"] = "1"
import numpy as np
import torch
import torch.nn.functional as F

MODEL = os.environ["MODEL"]
CKPT = os.environ.get("CKPT", "")
SPLIT = os.environ.get("SPLIT", "val")
N_SEQ = int(os.environ.get("N_SEQ", "0"))
BATCH = int(os.environ.get("BATCH", "8"))
assert SPLIT in ("val", "test"), SPLIT

KOLLAR = {
    "L":     ("meteonet_sp_gat_lagrange.py", None, {"MOTION_EDGES": "0"}),
    "E":     ("meteonet_sp_gat_lagrange.py", None, {"MOTION_EDGES": "0"}),
    "A":     ("meteonet_ana_gat_lagrange_residual.py", "MeteoNetGATLagrangeResidual", {"MOTION_EDGES": "0"}),
    "B":     ("meteonet_ana_gat_lagrange_residual.py", "MeteoNetGATLagrangeResidual", {"MOTION_EDGES": "1"}),
    "Bsham": ("meteonet_ana_gat_lagrange_residual.py", "MeteoNetGATLagrangeResidual",
              {"MOTION_EDGES": "1", "MOTION_SIGN": "-1"}),
    "K":     ("meteonet_ana_conv3x3_lagrange_residual.py", "MeteoNetConvLagrangeResidual", {}),
    "C":     ("meteonet_baseline_convlstm_lagrange.py", "MeteoNetConvLSTMLagrangeResidual", {}),
    "D":     ("meteonet_ana_gat_direct_multi.py", "MeteoNetGATDirectMulti", {}),
    "H":     ("meteonet_ana_gat_lagrange_hiz.py", "MeteoNetGATLagrangeHiz", {"MOTION_EDGES": "0"}),
    "G":     ("meteonet_ana_gat_lagrange_gecmis.py", "MeteoNetGATLagrangeGecmis", {"MOTION_EDGES": "0"}),
    "C0":    ("meteonet_baseline_convlstm.py", "MeteoNetConvLSTM", {}),
    "GAT_LSTM":     ("meteonet_ana_graphconvlstm_gat.py", "MeteoNetGraphConvLSTMGAT", {"ROLLOUT": "dup"}),
    "GCN_LSTM":     ("meteonet_ana_gcn_lstm.py", "MeteoNetGCNLSTM", {"ROLLOUT": "dup"}),
    "Conv3x3_LSTM": ("meteonet_ana_conv3x3_lstm.py", "MeteoNetConv3x3LSTM", {"ROLLOUT": "dup"}),
    "Att_LSTM":     ("meteonet_ana_transformer_attention.py", "MeteoNetTransformerLSTM", {"ROLLOUT": "dup"}),
}
OE_KOLLARI = {"GAT_LSTM", "GCN_LSTM", "Conv3x3_LSTM", "C0", "Att_LSTM"}
ALTKUME_IZINLI = OE_KOLLARI | {"A", "B", "Bsham", "K", "C", "D"}
ALTKUME_V1_SHA = "0c87f765ba693567ecd575dac5cf999bd01c5f7fd5d3b97c7d7dad3d13976b96"
_VS_VARSAYILAN = os.path.join(ROOT, "analysis", "split", "strict_split_selection.json")
if os.environ.get("VERI_SECIMI") and os.path.abspath(os.environ["VERI_SECIMI"]) != os.path.abspath(_VS_VARSAYILAN):
    raise SystemExit(f"DUR: VERI_SECIMI={os.environ['VERI_SECIMI']} kati secim degil; degerlendirme kati secimle yapilir. "
                     f"Ortam degiskenini kaldirin.")
os.environ.pop("VERI_SECIMI", None)
dosya, sinif, bayrak = KOLLAR[MODEL]
STRICT = os.environ.get("STRICT_SPLIT", "1") == "1"
os.environ["STRICT_SPLIT"] = "0"
for k in ("MOTION_EDGES", "MOTION_SIGN", "ROLLOUT"):
    os.environ.pop(k, None)
os.environ.update(bayrak)
spec = importlib.util.spec_from_file_location("kol", os.path.join(ROOT, dosya))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
THR = list(M.CSI_THRESHOLDS)
assert len(THR) == 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import hashlib


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


import uuid, socket, atexit
KOSU_ID = str(uuid.uuid4())

_ek = (f"_{os.path.splitext(os.path.basename(CKPT))[0]}" if CKPT else ("_egitimsiz" if sinif else "")) \
      + ("_kati" if STRICT else "") + (f"_N{N_SEQ}" if N_SEQ else "")
OUT = os.environ.get("OUT") or os.path.join(ROOT, "analysis", f"ortak_deg_{MODEL}_{SPLIT}{_ek}.json")
NPZ = OUT[:-5] + "_sayimlar.npz"
KILIT = OUT + ".kilit"
try:
    _fd = os.open(KILIT, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    raise SystemExit(f"cikti kilitli (baska bir surec yaziyor ya da onceki kosu yarida kaldi): {KILIT}\n"
                     f"  boyle bir surec yoksa kilit dosyasini elle silin")
os.write(_fd, json.dumps({"kosu_id": KOSU_ID, "pid": os.getpid(), "host": socket.gethostname(),
                          "zaman": time.strftime("%Y-%m-%dT%H:%M:%S%z")}).encode())
os.close(_fd)
atexit.register(lambda: os.path.exists(KILIT) and os.remove(KILIT))
if os.environ.get("OVERWRITE", "0") != "1":
    for _p in (OUT, NPZ):
        assert not os.path.exists(_p), f"cikti zaten var, uzerine yazilmaz (OVERWRITE=1 ile izin): {_p}"

KOL_AYAR = {"model": sinif, "model_tag": getattr(M, "MODEL_TAG", None),
            "motion_edges": getattr(M, "MOTION_EDGES", None), "motion_sign": getattr(M, "MOTION_SIGN", None),
            "horizon_gat": getattr(M, "HORIZON_GAT", None), "base": getattr(M, "BASE_TAG", None),
            "strict_split": STRICT}
if MODEL in ("H", "G"):
    KOL_AYAR["vmax_px"] = getattr(M, "VMAX_PX", None)
if MODEL in OE_KOLLARI:
    KOL_AYAR["rollout"] = getattr(M, "ROLLOUT", None)
_AB = ["model", "motion_edges", "motion_sign", "horizon_gat", "base", "strict_split"]
ZORUNLU = {"A": _AB, "B": _AB, "Bsham": _AB, "K": ["model", "horizon_gat", "base", "strict_split"],
           "C": ["model", "base", "strict_split"], "D": ["model", "horizon_gat", "strict_split"], "C0": ["model"],
           "H": ["model", "motion_edges", "horizon_gat", "base", "strict_split", "vmax_px"],
           "G": ["model", "motion_edges", "horizon_gat", "base", "strict_split", "vmax_px"],
           "GAT_LSTM": ["model", "strict_split", "rollout"], "GCN_LSTM": ["model", "strict_split", "rollout"],
           "Conv3x3_LSTM": ["model", "strict_split", "rollout"], "Att_LSTM": ["model", "strict_split", "rollout"]}
import veri_secimi as VS
VS_KIMLIK = VS.kimlik() if STRICT else None
ESKI = os.environ.get("ESKI_YAN_KAYIT", "0") == "1"
assert not ESKI or MODEL == "C0", "ESKI_YAN_KAYIT=1 yalniz MODEL=C0 (eski bildiri kosulari) icin"
ckpt_bilgi = None
net = None
if sinif is not None:
    torch.manual_seed(42)
    net = getattr(M, sinif)().to(DEVICE)
    if CKPT:
        net.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True), strict=True)
        yan = CKPT[:-3] + "_metrics.json" if CKPT.endswith(".pt") else None
        ckpt_bilgi = {"sha256": sha256(CKPT), "yan_kayit": None, "eski_uyumluluk": ESKI}
        if yan and os.path.exists(yan):
            with open(yan) as f:
                yk = json.load(f)
            zorunlu = ZORUNLU[MODEL] + (["rollout", "strict_split"] if MODEL == "C0" and not ESKI else [])
            eksik = [k for k in zorunlu + ([] if ESKI else ["ckpt_sha256", "veri_secimi"]) if k not in yk]
            assert not eksik, f"yan kayitta zorunlu alan eksik {eksik}: {yan}"
            for k in zorunlu:
                assert yk[k] == KOL_AYAR[k], f"checkpoint baska kolun: {k} yan kayitta {yk[k]}, MODEL={MODEL} icin {KOL_AYAR[k]}"
            if not ESKI:
                assert yk["ckpt_sha256"] == ckpt_bilgi["sha256"], (
                    f"yan kayit bu checkpoint'e ait degil (ckpt_sha256 farkli): {yan}")
                if STRICT:
                    _vs = yk["veri_secimi"]
                    _izinli = {VS_KIMLIK["sha256"]} | ({ALTKUME_V1_SHA} if MODEL in ALTKUME_IZINLI else set())
                    assert isinstance(_vs, dict) and _vs.get("sha256") in _izinli, (
                        f"checkpoint baska bir veri secimiyle egitilmis (veri_secimi.sha256 izinli degil ya da yok): {yan}")
                else:
                    assert yk["veri_secimi"] is None, f"checkpoint kati secimle egitilmis ama degerlendirme kati degil: {yan}"
            ckpt_bilgi.update({"yan_kayit": os.path.basename(yan), "seed": yk.get("seed"),
                               "model_tag": yk.get("model_tag"), "epochs_ran": yk.get("epochs_ran"),
                               "all_losses_finite": yk.get("all_losses_finite")})
            if (MODEL in ALTKUME_IZINLI or MODEL in ("H", "G")) and not ESKI:
                ckpt_bilgi.update({"egitim_veri_secimi_sha256": (yk.get("veri_secimi") or {}).get("sha256"),
                                   "rollout": yk.get("rollout"), "bmse_weights": yk.get("bmse_weights"),
                                   "n_seq_egitim": yk.get("n_seq"), "max_epochs": yk.get("max_epochs")})
        else:
            assert os.environ.get("CKPT_YAN_KAYIT_YOK_OK", "0") == "1", (
                f"checkpoint yan kaydi yok: {yan} -> kol bayraklari dogrulanamaz (CKPT_YAN_KAYIT_YOK_OK=1 ile gec)")
    net.eval()

veri = M.VAL_FILE if SPLIT == "val" else M.TEST_FILE
ds = M.MeteoNetDataset(veri, max_samples=None)
assert getattr(ds, "idx", None) is None, "model Dataset'i kendi secimini uygulamamali (STRICT_SPLIT=0 bekleniyordu)"
VERI_KIMLIK = getattr(ds, "veri_kimlik", None) if hasattr(ds, "flow_kimlik") else None
if STRICT and VERI_KIMLIK is None:
    VERI_KIMLIK = dict(VS.veri_dogrula(veri), kaynak="veri_tam_dogrulamasi")
if STRICT:
    _ks = VERI_KIMLIK["dogrulanmis_kaynak_sha256"]
    ORIJINAL = VS.secim(SPLIT, len(ds), kaynak_sha256=_ks)
else:
    ORIJINAL = np.arange(len(ds))
if N_SEQ:
    ORIJINAL = ORIJINAL[:N_SEQ]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ORIJINAL.tolist()), batch_size=BATCH,
                                 shuffle=False, num_workers=0)
N = int(ORIJINAL.size)
if STRICT and not N_SEQ:
    assert N == {"val": 2080, "test": 2114}[SPLIT], f"kati {SPLIT} {N} dizi; beklenen {'2080' if SPLIT == 'val' else '2114'}"
OLCEK = (("piksel", 1), ("pool4", 4), ("pool16", 16))
say = np.zeros((N, 6, 6, 3, 3), dtype=np.int64)
kutle = np.zeros((N, 6, 2), dtype=np.float64)

t0 = time.time()
i0 = 0
with torch.no_grad():
    for xb, yb, *ex in dl:
        if MODEL == "L":
            pred = ex[0]
        elif MODEL == "E":
            pred = xb[:, -1:].expand_as(yb)
        else:
            pred = net(xb.to(DEVICE), *[e.to(DEVICE) for e in ex]).float().cpu()
        assert torch.isfinite(pred).all(), f"tahminde NaN/Inf (dizi {i0}..{i0 + pred.shape[0] - 1})"
        assert torch.isfinite(yb).all() and all(torch.isfinite(e).all() for e in ex), (
            f"hedef/taban/akista NaN/Inf (dizi {i0}..)")
        pred = pred.clamp(0.0, 1.0)[:, :, 0]
        obs = yb[:, :, 0].float()
        b = pred.shape[0]
        kutle[i0:i0 + b, :, 0] = pred.double().sum(dim=(2, 3)).numpy()
        kutle[i0:i0 + b, :, 1] = obs.double().sum(dim=(2, 3)).numpy()
        for si, (_, p) in enumerate(OLCEK):
            pp = pred if p == 1 else F.max_pool2d(pred, p, p)
            oo = obs if p == 1 else F.max_pool2d(obs, p, p)
            for j, t in enumerate(THR):
                a, o = pp >= t, oo >= t
                say[i0:i0 + b, :, j, si, 0] = (a & o).sum(dim=(2, 3)).numpy()
                say[i0:i0 + b, :, j, si, 1] = (a & ~o).sum(dim=(2, 3)).numpy()
                say[i0:i0 + b, :, j, si, 2] = (~a & o).sum(dim=(2, 3)).numpy()
        i0 += b
assert i0 == N


def csi(c):
    return c[..., 0] / np.maximum(c[..., 0] + c[..., 1] + c[..., 2], 1)


toplam = say.sum(0)
ozet = {}
for si, (ad, _) in enumerate(OLCEK):
    genel = toplam[:, :, si].sum(0)
    T, Fp, Fn = genel[:, 0], genel[:, 1], genel[:, 2]
    ufuk = csi(toplam[:, :, si]).mean(1)
    ozet[ad] = {"csi_genel": float(csi(genel).mean()), "csi_esik": csi(genel).tolist(),
                "csi_ufuk": ufuk.tolist(), "bias_esik": ((T + Fp) / np.maximum(T + Fn, 1)).tolist(),
                "pod_esik": (T / np.maximum(T + Fn, 1)).tolist(), "far_esik": (Fp / np.maximum(T + Fp, 1)).tolist()}
pu = ozet["piksel"]["csi_ufuk"]
k_top = kutle.sum(0)
out = {
    "kosu_id": KOSU_ID,
    "model": MODEL, "dosya": dosya, "sinif": sinif, "bayraklar": bayrak, "kol_ayar": KOL_AYAR,
    "ckpt": CKPT or None, "ckpt_bilgi": ckpt_bilgi,
    "split": SPLIT, "N_seq": N, "strict_split": STRICT, "veri_secimi": VS_KIMLIK,
    "dizi_indeksleri": "ozgun dosya indeksleri NPZ'de (dizi_indeksleri); sira = sayim dizisinin ilk ekseni",
    "veri_dosyasi": {"ad": os.path.basename(veri), "boyut_bayt": os.path.getsize(veri)},
    "akis_kimligi": getattr(ds, "flow_kimlik", None),
    "gecmis_kimligi": getattr(ds, "gecmis_kimlik", None),
    "veri_kimligi": VERI_KIMLIK,
    "flow_method": getattr(M, "FLOW_METHOD", None),
    "S_long_t30_60": float(np.mean(pu[2:6])), "S_t40_60": float(np.mean(pu[3:6])),
    "csi_t10": float(pu[0]), "csi_at10mmh": float(ozet["piksel"]["csi_esik"][5]),
    "pool16_genel": ozet["pool16"]["csi_genel"], "piksel_genel": ozet["piksel"]["csi_genel"],
    "kutle_orani": float(k_top[:, 0].sum() / max(k_top[:, 1].sum(), 1e-12)),
    "kutle_orani_ufuk": (k_top[:, 0] / np.maximum(k_top[:, 1], 1e-12)).tolist(),
    "olcekler": ozet, "sure_s": round(time.time() - t0, 1),
}
_npz_tmp = f"{NPZ[:-4]}.{KOSU_ID}.tmp.npz"
np.savez_compressed(_npz_tmp, say=say, kutle=kutle, dizi_indeksleri=ORIJINAL,
                    kosu_kimligi=json.dumps({"kosu_id": KOSU_ID, "model": MODEL, "split": SPLIT, "ckpt_sha256":
                                             (ckpt_bilgi or {}).get("sha256"), "json": os.path.basename(OUT)}),
                    eksen="dizi,ufuk,esik,olcek(piksel,pool4,pool16),(TP,FP,FN)")
os.replace(_npz_tmp, NPZ)
_json_tmp = f"{OUT}.{KOSU_ID}.tmp"
with open(_json_tmp, "w") as f:
    json.dump(out, f, indent=1)
os.replace(_json_tmp, OUT)
print(json.dumps({k: v for k, v in out.items() if k != "olcekler"}, indent=1))
print("yazildi:", OUT)
