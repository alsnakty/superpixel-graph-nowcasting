# ConvLSTM reference: two layers, 64 channels, 3x3 kernels, full resolution, autoregressive.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
USE_AMP    = os.environ.get("USE_AMP", "1") == "1"
ROLLOUT    = os.environ.get("ROLLOUT", "dup")
EVAL_TEST  = os.environ.get("EVAL_TEST", "1") == "1"
STRICT_SPLIT = os.environ.get("STRICT_SPLIT", "0") == "1"
if STRICT_SPLIT:
    import sys as _sys
    _sys.path.insert(0, os.environ.get("LK_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")))
    import veri_secimi as VS
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "100"))

# settings

SEED      = int(os.environ.get("SEED", "42"))
MODEL_TAG = "baseline_convlstm" + os.environ.get("TAG_SUFFIX", "")

DATA_DIR  = os.environ.get("DATA_DIR", "./data")
OUT_DIR   = os.environ.get("OUT_DIR",  "./outputs")
FIG_DIR   = os.environ.get(
    "FIG_DIR",
    os.path.join("./figures", f"{MODEL_TAG}_seed{SEED}"))

TRAIN_FILE = os.path.join(DATA_DIR, "meteonet_nw_train.npy")
VAL_FILE   = os.path.join(DATA_DIR, "meteonet_nw_val.npy")
TEST_FILE  = os.path.join(DATA_DIR, "meteonet_nw_test.npy")

MODEL_SAVE_PATH = os.path.join(OUT_DIR, f"{MODEL_TAG}_seed{SEED}.pt")
METRICS_PATH    = os.path.join(OUT_DIR, f"{MODEL_TAG}_seed{SEED}_metrics.json")

N_INPUT    = 7
N_PREDICT  = 6
IMG_SIZE   = 128

HIDDEN_DIM  = 64
N_LAYERS    = 2
KERNEL_SIZE = 3

EPOCHS     = 1 if SMOKE_TEST else MAX_EPOCHS
PATIENCE   = 15
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
LR         = 1e-3
BATCH_SIZE = 4

# thresholds 0.1, 0.5, 1, 2, 5, 10 mm/h divided by the 12 mm/h cap
CSI_THRESHOLDS = [0.1/12, 0.5/12, 1.0/12, 2.0/12, 5.0/12, 10.0/12]
# plain MSE: BMSE_WEIGHTS="1,1,1,1,1,1,1"
BMSE_WEIGHTS   = [float(x) for x in
                  os.environ.get("BMSE_WEIGHTS", "1,2,5,10,15,20,30").split(",")]
assert len(BMSE_WEIGHTS) == 7, (
    f"BMSE_WEIGHTS 7 deger olmali, {len(BMSE_WEIGHTS)} geldi: {BMSE_WEIGHTS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
if SMOKE_TEST:
    print(">>> SMOKE TEST modu aktif (1 epoch, 2 batch) <<<")
print(f"MODEL_TAG: {MODEL_TAG}  |  SEED: {SEED}  |  USE_AMP: {int(USE_AMP)}"
      f"  |  ROLLOUT: {ROLLOUT}")
print(f"Çıktı: {MODEL_SAVE_PATH}")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# colour map

def get_radar_cmap():
    colors = ['white', '#8ec7ff', 'dodgerblue', 'lime',
              'yellow', 'darkorange', 'red', 'darkmagenta']
    return col.LinearSegmentedColormap.from_list('radar', colors)

CMAP = get_radar_cmap()

# data

class MeteoNetDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, max_samples=None):
        print(f"Yükleniyor (memmap): {filepath}")
        self.data = np.load(filepath, mmap_mode='r')
        self.idx = None
        self.veri_kimlik = None
        if STRICT_SPLIT:
            self.veri_kimlik = dict(VS.veri_dogrula(filepath), kaynak="veri_tam_dogrulamasi")
            self.idx = VS.secim(VS.split_adi(filepath), self.data.shape[0],
                                kaynak_sha256=self.veri_kimlik["dogrulanmis_kaynak_sha256"])
            if max_samples is not None:
                self.idx = self.idx[:max_samples]
            self.n = int(self.idx.size)
        else:
            if max_samples is not None:
                self.data = self.data[:max_samples]
            self.n = self.data.shape[0]
        print(f"  {self.n} örnek  {self.data.shape}  STRICT_SPLIT={int(STRICT_SPLIT)}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.idx is not None:
            idx = int(self.idx[idx])
        seq = self.data[idx].astype(np.float32)
        seq = seq[::2]
        x = torch.from_numpy(seq[:N_INPUT]).unsqueeze(1)
        y = torch.from_numpy(seq[N_INPUT:]).unsqueeze(1)
        return x, y

# ConvLSTM cell

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        # gates i, f, g, o from concat(x, h); no peephole, no residual
        self.conv = nn.Conv2d(in_channels + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding)

        # forget-gate bias = 1
        with torch.no_grad():
            self.conv.bias[hidden_dim:2 * hidden_dim].fill_(1.0)

    def forward(self, x, h, c):
        z = torch.cat([x, h], dim=1)              # (B, in+hidden, H, W)
        gates = self.conv(z)                      # (B, 4*hidden, H, W)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# model

class MeteoNetConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.cells = nn.ModuleList([
            ConvLSTMCell(
                in_channels = in_channels if i == 0 else hidden_dim,
                hidden_dim  = hidden_dim,
                kernel_size = kernel_size,
            )
            for i in range(n_layers)
        ])
        # 1x1 convolution: hidden -> 1 channel
        self.out_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def _step(self, frame, h_list, c_list):
        x = frame
        new_h, new_c = [], []
        for i, cell in enumerate(self.cells):
            h, c = cell(x, h_list[i], c_list[i])
            new_h.append(h); new_c.append(c)
            x = h
        return new_h, new_c

    def forward(self, x_in):
        B, T, C, H, W = x_in.shape

        h_list = [torch.zeros(B, self.hidden_dim, H, W, device=x_in.device)
                  for _ in range(self.n_layers)]
        c_list = [torch.zeros(B, self.hidden_dim, H, W, device=x_in.device)
                  for _ in range(self.n_layers)]

        T_enc = T - 1 if ROLLOUT == "single" else T
        for t in range(T_enc):
            h_list, c_list = self._step(x_in[:, t], h_list, c_list)

        # autoregressive: each forecast is fed back as the next input
        pred_frame = x_in[:, -1]
        outputs = []
        for _ in range(N_PREDICT):
            h_list, c_list = self._step(pred_frame, h_list, c_list)
            pred_raw = self.out_proj(h_list[-1])         # raw output, no sigmoid
            outputs.append(pred_raw.unsqueeze(1))
            pred_frame = pred_raw.clamp(0, 1)

        return torch.cat(outputs, dim=1)

# loss

def bmse_loss(pred, target):
    w = torch.ones_like(target)
    w = torch.where(target >= 0.1/12,  torch.full_like(w, BMSE_WEIGHTS[1]), w)
    w = torch.where(target >= 0.5/12,  torch.full_like(w, BMSE_WEIGHTS[2]), w)
    w = torch.where(target >= 1.0/12,  torch.full_like(w, BMSE_WEIGHTS[3]), w)
    w = torch.where(target >= 2.0/12,  torch.full_like(w, BMSE_WEIGHTS[4]), w)
    w = torch.where(target >= 5.0/12,  torch.full_like(w, BMSE_WEIGHTS[5]), w)
    w = torch.where(target >= 10.0/12, torch.full_like(w, BMSE_WEIGHTS[6]), w)
    return (w * (pred - target) ** 2).mean()

# metrics

def compute_csi(pred_np, true_np, threshold):
    p, t = pred_np >= threshold, true_np >= threshold
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    d = tp + fp + fn
    return float(tp / d) if d > 0 else 0.0

def _max_pool_np(arr, pool_size):
    t = torch.from_numpy(arr).float().unsqueeze(1)
    pooled = F.max_pool2d(t, kernel_size=pool_size, stride=pool_size)
    return pooled.squeeze(1).numpy()

def evaluate_metrics(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)
            all_pred.append(pred[:, :, 0])
            all_true.append(yb.numpy()[:, :, 0])
    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    N, T_pred, H, W = pred_all.shape

    ssim_matrix = np.zeros((N, T_pred), dtype=np.float64)
    for i in range(N):
        for t in range(T_pred):
            ssim_matrix[i, t] = sk_ssim(pred_all[i, t], true_all[i, t], data_range=1.0)

    per_lt_mse, per_lt_ssim, per_lt_csi_m, per_lt_csi_per_thr = [], [], [], []
    for t in range(T_pred):
        p_t = pred_all[:, t]; y_t = true_all[:, t]
        mse_t = float(np.mean((p_t - y_t) ** 2))
        per_lt_mse.append(mse_t)
        per_lt_ssim.append(float(ssim_matrix[:, t].mean()))
        csi_t = [compute_csi(p_t, y_t, thr) for thr in CSI_THRESHOLDS]
        per_lt_csi_per_thr.append([round(c, 4) for c in csi_t])
        per_lt_csi_m.append(float(np.mean(csi_t)))

    pred_flat = pred_all.reshape(-1, H, W)
    true_flat = true_all.reshape(-1, H, W)
    csi_scores  = [compute_csi(pred_flat, true_flat, thr) for thr in CSI_THRESHOLDS]
    pred_p4,  true_p4  = _max_pool_np(pred_flat, 4),  _max_pool_np(true_flat, 4)
    pred_p16, true_p16 = _max_pool_np(pred_flat, 16), _max_pool_np(true_flat, 16)
    csi_pool4   = [compute_csi(pred_p4,  true_p4,  thr) for thr in CSI_THRESHOLDS]
    csi_pool16  = [compute_csi(pred_p16, true_p16, thr) for thr in CSI_THRESHOLDS]
    mse = float(np.mean((pred_all - true_all) ** 2))

    return {
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "ssim": float(ssim_matrix.mean()),
        "csi_m": float(np.mean(csi_scores)),
        "csi_per_threshold": [round(c, 4) for c in csi_scores],
        "csi_pool4_m":  float(np.mean(csi_pool4)),
        "csi_pool4_per_threshold":  [round(c, 4) for c in csi_pool4],
        "csi_pool16_m": float(np.mean(csi_pool16)),
        "csi_pool16_per_threshold": [round(c, 4) for c in csi_pool16],
        "per_leadtime": {
            "mse":  [round(float(m), 6) for m in per_lt_mse],
            "rmse": [round(float(np.sqrt(m)), 6) for m in per_lt_mse],
            "ssim": [round(float(s), 4) for s in per_lt_ssim],
            "csi_m": [round(float(c), 4) for c in per_lt_csi_m],
            "csi_per_threshold": per_lt_csi_per_thr,
        },
    }

# training

def train_model(model, train_loader, val_loader):
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion  = bmse_loss
    scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and USE_AMP))
    warmup_steps = max(1, WARMUP_EPOCHS * len(train_loader))
    global_step  = 0
    best_val   = float('inf')
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [],
                  "train_sqrt_bmse": [], "val_sqrt_bmse": [],
                  "lr": [], "grad_norm": [], "grad_norm_inf": [], "amp_scale": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = __import__('time').time()
        model.train()
        train_loss = 0.0; n_batches = 0; n_train_seen = 0
        gn_sum = 0.0; gn_n = 0; gn_inf = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if global_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg['lr'] = LR * (global_step + 1) / warmup_steps
            global_step += 1
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if np.isfinite(gn_val):
                gn_sum += gn_val
                gn_n   += 1
            else:
                gn_inf += 1
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * xb.size(0)
            n_train_seen += xb.size(0); n_batches += 1
            if SMOKE_TEST and n_batches >= 2:
                print(f"  Smoke test: {n_batches} batch tamamlandı, forward/backward OK")
                break
        train_loss /= max(n_train_seen, 1)
        epoch_lr    = optimizer.param_groups[0]['lr']
        epoch_gn    = gn_sum / max(gn_n, 1)
        epoch_gninf = gn_inf
        epoch_scale = scaler.get_scale()

        model.eval()
        val_loss = 0.0; n_batches = 0; n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
                n_val_seen += xb.size(0); n_batches += 1
                if SMOKE_TEST and n_batches >= 2:
                    break
        val_loss /= max(n_val_seen, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sqrt_bmse"].append(float(np.sqrt(train_loss)))
        history["val_sqrt_bmse"].append(float(np.sqrt(val_loss)))
        history["lr"].append(float(epoch_lr))
        history["grad_norm"].append(float(epoch_gn))
        history["grad_norm_inf"].append(int(epoch_gninf))
        history["amp_scale"].append(float(epoch_scale))
        if epoch > WARMUP_EPOCHS:
            scheduler.step(val_loss)

        elapsed = __import__('time').time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train BMSE: {train_loss:.5f}  sqrt: {np.sqrt(train_loss):.4f}  |  "
              f"Val BMSE: {val_loss:.5f}  sqrt: {np.sqrt(val_loss):.4f}  "
              f"| lr: {epoch_lr:.2e}  gn: {epoch_gn:.3f}  gninf: {epoch_gninf:d}  "
              f"amp: {epoch_scale:.0f}  "
              f"[{elapsed:.0f}s]")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"EarlyStopping tetiklendi ({PATIENCE} epoch).")
                break

    print(f"En iyi Val BMSE: {best_val:.5f}")
    return history

# figures

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train BMSE")
    axes[0].plot(history["val_loss"],   label="Val BMSE")
    axes[0].set_title("B-MSE Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_sqrt_bmse"], label="Train sqrt(BMSE)")
    axes[1].plot(history["val_sqrt_bmse"],   label="Val sqrt(BMSE)")
    axes[1].set_title("sqrt(B-MSE)"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_history.png"), dpi=150)
    plt.close()

def save_comparison_gif(true_frames, pred_frames, fps=3):
    frames = []
    for i in range(len(true_frames)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(true_frames[i], cmap=CMAP, vmin=0, vmax=1)
        axes[0].set_title(f"Gerçek t+{(i+1)*10}dk"); axes[0].axis("off")
        axes[1].imshow(pred_frames[i], cmap=CMAP, vmin=0, vmax=1)
        axes[1].set_title(f"Tahmin t+{(i+1)*10}dk"); axes[1].axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    path = os.path.join(FIG_DIR, "meteonet_baseline_convlstm_comparison.gif")
    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF kaydedildi: {path}")

# main

def _seed_worker(worker_id):
    ws = SEED + worker_id
    np.random.seed(ws); random.seed(ws)


if __name__ == "__main__":
    assert os.path.exists(TRAIN_FILE), f"Veri bulunamadı: {TRAIN_FILE}"
    assert os.path.exists(VAL_FILE),   f"Veri bulunamadı: {VAL_FILE}"
    if not SMOKE_TEST and EVAL_TEST:
        assert os.path.exists(TEST_FILE), f"Veri bulunamadı: {TEST_FILE}"

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator(); g.manual_seed(SEED)

    max_samples = 16 if SMOKE_TEST else None
    train_ds = MeteoNetDataset(TRAIN_FILE, max_samples=max_samples)
    val_ds   = MeteoNetDataset(VAL_FILE,   max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              worker_init_fn=_seed_worker, generator=g)

    if not SMOKE_TEST and EVAL_TEST:
        test_ds  = MeteoNetDataset(TEST_FILE, max_samples=max_samples)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  worker_init_fn=_seed_worker, generator=g)

    _xb, _yb = next(iter(train_loader))
    print(f"Batch shape kontrolü — x: {tuple(_xb.shape)}  y: {tuple(_yb.shape)}")
    assert _xb.shape[1:] == (N_INPUT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik x shape: {tuple(_xb.shape)}")
    assert _yb.shape[1:] == (N_PREDICT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik y shape: {tuple(_yb.shape)}")
    del _xb, _yb

    model = MeteoNetConvLSTM(
        in_channels = 1,
        hidden_dim  = HIDDEN_DIM,
        n_layers    = N_LAYERS,
        kernel_size = KERNEL_SIZE,
    ).to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total:,}")
    print(f"MeteoNet vanilla ConvLSTM baseline (peephole'suz, 2-layer stacked, encoder yok)")
    print(f"BMSE_WEIGHTS: {BMSE_WEIGHTS}")
    print(f"HIDDEN_DIM: {HIDDEN_DIM} | N_LAYERS: {N_LAYERS} | KERNEL_SIZE: {KERNEL_SIZE}\n")

    history = train_model(model, train_loader, val_loader)

    if SMOKE_TEST:
        print("\n>>> SMOKE TEST BAŞARILI — ortam doğrulandı <<<")
        if DEVICE.type == "cuda":
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA OK, ConvLSTM OK, AMP OK, forward/backward OK")
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        print("\nMetrikler hesaplanıyor (val)...")
        val_metrics  = evaluate_metrics(model, val_loader)
        if EVAL_TEST:
            print("Metrikler hesaplanıyor (test)...")
            test_metrics = evaluate_metrics(model, test_loader)
        else:
            print("EVAL_TEST=0: test kumesi degerlendirilmedi (protokol: test en sonda bir kez).")
            test_metrics = None

        out = {
            "model":          "MeteoNetConvLSTM",
            "model_tag":      MODEL_TAG,
            "operator":       "vanilla ConvLSTM (peephole'suz, 3x3 conv cell, encoder yok)",
            "seed":           SEED,
            "loss":           "B-MSE",
            "bmse_weights":   BMSE_WEIGHTS,
            "epochs_ran":     len(history["train_loss"]),
            "best_val_bmse":  round(min(history["val_loss"]), 6),
            "warmup_epochs":  WARMUP_EPOCHS,
            "patience":       PATIENCE,
            "base_lr":        LR,
            "batch_size":     BATCH_SIZE,
            "tag_suffix":     os.environ.get("TAG_SUFFIX", ""),
            "use_amp":        USE_AMP,
            "rollout":        ROLLOUT,
            "lr_end_of_epoch":     history["lr"],
            "grad_norm_per_epoch": history["grad_norm"],
            "grad_norm_inf_per_epoch": history["grad_norm_inf"],
            "amp_scale_per_epoch": history["amp_scale"],
            "eval_test":      EVAL_TEST,
            "strict_split":   STRICT_SPLIT,
            "veri_secimi":    (VS.kimlik() if STRICT_SPLIT else None),
            "n_seq":          {"train": len(train_ds), "val": len(val_ds)},
            "max_epochs":     EPOCHS,
            "num_workers":    NUM_WORKERS,
            "all_losses_finite": bool(np.all(np.isfinite(history["train_loss"]))
                                      and np.all(np.isfinite(history["val_loss"]))),
            "ckpt_sha256":    __import__("hashlib").sha256(open(MODEL_SAVE_PATH, "rb").read()).hexdigest(),
            "veri_kimligi":   {sp: dict({"ad": os.path.basename(fp), "boyut_bayt": os.path.getsize(fp)},
                                        **(ds.veri_kimlik or {}))
                               for sp, fp, ds in (("train", TRAIN_FILE, train_ds), ("val", VAL_FILE, val_ds))},
            "val":  val_metrics,
            "test": test_metrics,
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n--- MeteoNet ConvLSTM baseline Sonuçlar ---")
        print(f"  Best Val BMSE : {out['best_val_bmse']:.6f}")
        print(f"  Epoch         : {out['epochs_ran']}")
        print(f"  -- Validation --")
        print(f"  MSE: {val_metrics['mse']:.6f}  RMSE: {val_metrics['rmse']:.6f}  "
              f"SSIM: {val_metrics['ssim']:.4f}  CSI-M: {val_metrics['csi_m']:.4f}")
        print(f"  CSI-pool4: {val_metrics['csi_pool4_m']:.4f}  "
              f"CSI-pool16: {val_metrics['csi_pool16_m']:.4f}")
        if test_metrics is not None:
            print(f"  -- Test --")
            print(f"  MSE: {test_metrics['mse']:.6f}  RMSE: {test_metrics['rmse']:.6f}  "
                  f"SSIM: {test_metrics['ssim']:.4f}  CSI-M: {test_metrics['csi_m']:.4f}")
            print(f"  CSI-pool4: {test_metrics['csi_pool4_m']:.4f}  "
                  f"CSI-pool16: {test_metrics['csi_pool16_m']:.4f}")

        plot_training_history(history)

        sample_x, sample_y = val_ds[0]
        model.eval()
        with torch.no_grad():
            pred = model(sample_x.unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)
        save_comparison_gif(sample_y[:, 0].numpy(), pred[:, 0])
        print(f"\nTamamlandı.\n  Model: {MODEL_SAVE_PATH}\n  Figürler: {FIG_DIR}")
