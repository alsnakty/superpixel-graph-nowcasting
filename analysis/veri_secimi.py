# Sequence selection for the time-separated training, validation and test sets.

import os, json, hashlib
import numpy as np

VARSAYILAN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "split", "strict_split_selection.json")


def _yol():
    return os.environ.get("VERI_SECIMI", VARSAYILAN)


def _yukle():
    p = _yol()
    if not os.path.exists(p):
        raise FileNotFoundError(f"Veri secim dosyasi yok: {p}\n  (analysis/split/ "
                                f"klasoru ya da en azindan strict_split_selection.json bulunmali)")
    with open(p) as f:
        return json.load(f)


def secim(split, n_dosya, kaynak_sha256=None):
    s = _yukle()
    sp = s["splits"][split]
    assert sp["original_count"] == n_dosya, (
        f"secim dosyasi baska veri surumu icin: {split} original_count {sp['original_count']} vs dosya {n_dosya}")
    if kaynak_sha256 is not None:
        beklenen = s["source_dataset_identities"][split]["sha256"]
        assert kaynak_sha256 == beklenen, f"secim dosyasi baska veri icin: {split} sha256 {beklenen[:16]}... vs {kaynak_sha256[:16]}..."
    idx = np.asarray(sp["retained_indices"], dtype=np.int64)
    assert idx.size == sp["retained_count"] and np.all(np.diff(idx) > 0) and idx.min() >= 0 and idx.max() < n_dosya
    assert not set(sp["excluded_indices"]) & set(idx.tolist())
    return idx


def kimlik():
    p = _yol()
    with open(p, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    return {"dosya": os.path.basename(p), "sha256": h, "politika": _yukle().get("policy")}


def _sha256(path, blok=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(blok), b""):
            h.update(b)
    return h.hexdigest()


def _durum(path):
    st = os.stat(path)
    return {"mutlak_yol": os.path.abspath(path), "boyut_bayt": st.st_size, "mtime_ns": st.st_mtime_ns}


def veri_dogrula(data_path):
    import socket, time
    rp = data_path + ".dogrulama.json"
    if os.environ.get("VERI_HASH_DOGRULA", "0") == "1":
        rec = {"kaynak_sha256": _sha256(data_path), "kaynak": _durum(data_path),
               "host": socket.gethostname(), "zaman": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        tmp = f"{rp}.{os.getpid()}.{time.time_ns()}.tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, rp)
        bu_kosuda = True
    else:
        if not os.path.exists(rp):
            raise FileNotFoundError(
                f"Veri tam dogrulama kaydi yok: {rp}\n  Bu makinede bir kez VERI_HASH_DOGRULA=1 ile calistirin "
                f"(ornegin smoke isinde) ya da: VERI_HASH_DOGRULA=1 python analysis/veri_secimi.py")
        with open(rp) as f:
            rec = json.load(f)
        assert rec["kaynak"] == _durum(data_path), (
            f"veri dosyasi dogrulamadan sonra degismis/tasinmis (yol, boyut ya da mtime farkli): {data_path}\n"
            f"  -> VERI_HASH_DOGRULA=1 ile yeniden dogrulayin")
        bu_kosuda = False
    return {"dogrulanmis_kaynak_sha256": rec["kaynak_sha256"], "dogrulama_zamani": rec["zaman"],
            "dogrulama_host": rec["host"], "bu_kosuda_hash": bu_kosuda}


def split_adi(dosya_yolu):
    ad = os.path.basename(dosya_yolu)
    for sp in ("train", "val", "test"):
        if ad == f"meteonet_nw_{sp}.npy":
            return sp
    raise ValueError(f"split adi cikarilamadi: {ad}")


if __name__ == "__main__":
    import sys
    data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    os.environ["VERI_HASH_DOGRULA"] = "1"
    s = _yukle()
    for sp in ("train", "val", "test"):
        dp = os.path.join(data_dir, f"meteonet_nw_{sp}.npy")
        k = veri_dogrula(dp)
        es = k["dogrulanmis_kaynak_sha256"] == s["source_dataset_identities"][sp]["sha256"]
        print(f"{sp}: {k['dogrulanmis_kaynak_sha256'][:16]}... secim dosyasiyla esles: {es} -> {dp}.dogrulama.json",
              flush=True)
        if not es:
            sys.exit(1)
