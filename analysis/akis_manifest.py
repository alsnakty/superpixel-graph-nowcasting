# Identity records that tie a motion-field file to the data file it was computed from.

import os, sys, json, time, socket, hashlib

KARELER = "13'luk dizi (ham[::2]) 5 -> 6; v = flow(onceki=seq[5], son=seq[6]); ham kare 10 -> 12"


def sha256(path, blok=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(blok), b""):
            h.update(b)
    return h.hexdigest()


def manifest_yolu(flow_path):
    return flow_path + ".manifest.json"


def dogrulama_yolu(flow_path):
    return flow_path + ".dogrulama.json"


def _atomik_json(path, obj):
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _dosya_durumu(path):
    st = os.stat(path)
    return {"mutlak_yol": os.path.abspath(path), "boyut_bayt": st.st_size, "mtime_ns": st.st_mtime_ns}


def olustur(data_path, flow_path, method, kayit_turu, uretim_denetimi=None):
    import numpy as np, scipy, skimage
    assert kayit_turu in ("uretim", "sonradan"), kayit_turu
    fl = np.load(flow_path, mmap_mode="r")
    src = np.load(data_path, mmap_mode="r")
    assert fl.shape[0] == src.shape[0], (fl.shape, src.shape)
    m = {
        "akis_dosyasi": os.path.basename(flow_path),
        "akis_boyut_bayt": os.path.getsize(flow_path),
        "akis_sekil": list(fl.shape),
        "akis_dtype": str(fl.dtype),
        "akis_sha256": sha256(flow_path),
        "kaynak_dosya": os.path.basename(data_path),
        "kaynak_boyut_bayt": os.path.getsize(data_path),
        "kaynak_sekil": list(src.shape),
        "kaynak_sha256": sha256(data_path),
        "yontem": method,
        "kareler": KARELER,
        "kayit_turu": kayit_turu,
        "uretim_denetimi": uretim_denetimi,
        "surumler": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-image": skimage.__version__},
        "surumler_notu": ("akisi ureten ortam" if kayit_turu == "uretim"
                          else "manifesti yazan ortam; akisi ureten ortami KANITLAMAZ"),
        "olusturma": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomik_json(manifest_yolu(flow_path), m)
    return m


def _manifest_kontrol(data_path, flow_path, method, n_seq):
    mp = manifest_yolu(flow_path)
    if not os.path.exists(mp):
        raise FileNotFoundError(
            f"Akis manifesti yok: {mp}\n  Olusturmak icin: python analysis/akis_manifest.py {method}\n"
            f"  (manifest dosyasi akis dosyasinin yaninda olmali.)")
    with open(mp) as f:
        m = json.load(f)
    assert m["yontem"] == method, f"akis yontemi uyusmuyor: manifest {m['yontem']} vs istenen {method}"
    assert m["kaynak_dosya"] == os.path.basename(data_path), (
        f"akis baska veri dosyasindan: {m['kaynak_dosya']} vs {os.path.basename(data_path)}")
    assert m["kaynak_boyut_bayt"] == os.path.getsize(data_path), "kaynak veri dosyasinin boyutu degismis"
    assert m["akis_boyut_bayt"] == os.path.getsize(flow_path), "akis dosyasinin boyutu manifestten farkli"
    assert m["akis_sekil"][0] == n_seq == m["kaynak_sekil"][0], (
        f"dizi sayisi uyusmuyor: akis {m['akis_sekil'][0]}, kaynak {m['kaynak_sekil'][0]}, Dataset {n_seq}")
    return m


def tam_dogrula(data_path, flow_path, m, n_ornek=8):
    import numpy as np, skimage
    t0 = time.time()
    a_sha, k_sha = sha256(flow_path), sha256(data_path)
    assert a_sha == m["akis_sha256"], f"akis dosyasinin SHA-256'si manifestten farkli: {flow_path}"
    assert k_sha == m["kaynak_sha256"], f"kaynak veri dosyasinin SHA-256'si manifestten farkli: {data_path}"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lagrange_kalicilik as LK
    eski = LK.FLOW
    LK.FLOW = m["yontem"]
    try:
        src = np.load(data_path, mmap_mode="r")
        fl = np.load(flow_path, mmap_mode="r")
        idx = np.random.default_rng(0).choice(src.shape[0], size=min(n_ornek, src.shape[0]), replace=False)
        farklar = []
        for i in idx:
            seq = np.asarray(src[i], dtype=np.float32)[::2]
            farklar.append(float(np.abs(np.asarray(fl[i]) - LK.flow(seq[5], seq[6])).max()))
    finally:
        LK.FLOW = eski
    ayni_surum = skimage.__version__ == m["surumler"]["scikit-image"]
    bit_ayni = sum(f == 0.0 for f in farklar)
    if ayni_surum:
        assert bit_ayni == len(farklar), (
            f"ayni scikit-image surumunde akis yeniden hesabi farkli ({bit_ayni}/{len(farklar)} bit-ayni, "
            f"maks fark {max(farklar):.3g}): akis bu kaynaktan uretilmemis olabilir")
    rec = {
        "akis_sha256": a_sha, "kaynak_sha256": k_sha, "manifest_akis_sha256": m["akis_sha256"],
        "akis": _dosya_durumu(flow_path), "kaynak": _dosya_durumu(data_path),
        "ornek_yeniden_hesap": {"indeksler": [int(i) for i in idx], "maks_fark": farklar,
                                "bit_ayni": f"{bit_ayni}/{len(farklar)}", "scikit_image": skimage.__version__,
                                "manifest_scikit_image": m["surumler"]["scikit-image"],
                                "not": ("ayni surum: bit-aynilik zorunlu tutuldu" if ayni_surum else
                                        "surum farkli: fark yalniz kaydedildi; kimlik SHA-256 ile kurulur")},
        "host": socket.gethostname(), "zaman": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sure_s": round(time.time() - t0, 1),
    }
    _atomik_json(dogrulama_yolu(flow_path), rec)
    return rec


def dogrula(data_path, flow_path, method, n_seq):
    m = _manifest_kontrol(data_path, flow_path, method, n_seq)
    if os.environ.get("AKIS_HASH_DOGRULA", "0") == "1":
        rec = tam_dogrula(data_path, flow_path, m)
        bu_kosuda = True
        print(f"  akis tam dogrulama: SHA-256 + ornek yeniden hesap {rec['ornek_yeniden_hesap']['bit_ayni']} "
              f"({os.path.basename(flow_path)}, {rec['sure_s']} s)")
    else:
        rp = dogrulama_yolu(flow_path)
        if not os.path.exists(rp):
            raise FileNotFoundError(
                f"Akis tam dogrulama kaydi yok: {rp}\n  Bu makinede bir kez AKIS_HASH_DOGRULA=1 ile calistirin "
                f"(ornegin smoke isinde) ya da: AKIS_HASH_DOGRULA=1 python analysis/akis_manifest.py --dogrula {method}")
        with open(rp) as f:
            rec = json.load(f)
        assert rec["akis_sha256"] == m["akis_sha256"] and rec["kaynak_sha256"] == m["kaynak_sha256"], (
            "dogrulama kaydi bu manifestle eslesmiyor -> AKIS_HASH_DOGRULA=1 ile yeniden dogrulayin")
        for ad, p in (("akis", flow_path), ("kaynak", data_path)):
            assert rec[ad] == _dosya_durumu(p), (
                f"{ad} dosyasi dogrulamadan sonra degismis/tasinmis (yol, boyut ya da mtime farkli): {p}\n"
                f"  -> AKIS_HASH_DOGRULA=1 ile yeniden dogrulayin")
        bu_kosuda = False
    kimlik = {"akis_dosyasi": m["akis_dosyasi"], "yontem": m["yontem"], "kayit_turu": m.get("kayit_turu"),
              "manifest_akis_sha256": m["akis_sha256"], "manifest_kaynak_sha256": m["kaynak_sha256"],
              "dogrulanmis_akis_sha256": rec["akis_sha256"], "dogrulanmis_kaynak_sha256": rec["kaynak_sha256"],
              "dogrulama_zamani": rec["zaman"], "dogrulama_host": rec["host"],
              "ornek_yeniden_hesap": rec["ornek_yeniden_hesap"]["bit_ayni"], "bu_kosuda_hash": bu_kosuda}
    return m, kimlik


if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    flow_dir = os.environ.get("FLOW_DIR", data_dir)
    args = sys.argv[1:]
    dogrulama = "--dogrula" in args
    for method in ([a for a in args if a != "--dogrula"] or ["tvl1"]):
        for split in ("train", "val", "test"):
            dp = os.path.join(data_dir, f"meteonet_nw_{split}.npy")
            fp = os.path.join(flow_dir, f"meteonet_nw_{split}_akis56_{method}.npy")
            if not os.path.exists(fp):
                print("yok, atlandi:", fp)
                continue
            t0 = time.time()
            if dogrulama:
                os.environ["AKIS_HASH_DOGRULA"] = "1"
                import numpy as np
                _, k = dogrula(dp, fp, method, int(np.load(fp, mmap_mode="r").shape[0]))
                print(f"{split}/{method}: dogrulandi {k['dogrulanmis_akis_sha256'][:16]}... "
                      f"ornek {k['ornek_yeniden_hesap']} -> {dogrulama_yolu(fp)}", flush=True)
            else:
                m = olustur(dp, fp, method, "sonradan")
                print(f"{split}/{method}: akis {m['akis_sha256'][:16]}... kaynak {m['kaynak_sha256'][:16]}... "
                      f"{time.time() - t0:.0f} s -> {manifest_yolu(fp)}", flush=True)
