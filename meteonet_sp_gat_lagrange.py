# LSP-GAT: Lagrangian persistence plus a correction learned on a superpixel graph.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from skimage.metrics import structural_similarity as sk_ssim
from skimage.segmentation import slic

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
SELF_TEST  = os.environ.get("SELF_TEST", "0") == "1"
USE_AMP    = os.environ.get("USE_AMP", "1") == "1"
PREDICTION_MODE = "direct_multi_horizon"
HORIZON_GAT = os.environ.get("HORIZON_GAT", "1") == "1"
import sys as _sys
_LK_DIR = os.environ.get("LK_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
_sys.path.insert(0, _LK_DIR)
import lagrange_kalicilik as LK
import akis_manifest as AM
import veri_secimi as VS
BASE_TAG = "lagrange_" + os.environ.get("FLOW_METHOD", "tvl1") + "_5to6"
MOTION_EDGES = os.environ.get("MOTION_EDGES", "0") == "1"
assert not (MOTION_EDGES and not HORIZON_GAT), "MOTION_EDGES=1 icin HORIZON_GAT=1 gerekir"
MOTION_SIGN = int(os.environ.get("MOTION_SIGN", "1"))
assert MOTION_SIGN in (1, -1), f"MOTION_SIGN 1 ya da -1 olmali: {MOTION_SIGN}"
assert MOTION_SIGN == 1 or MOTION_EDGES, "MOTION_SIGN=-1 yalniz MOTION_EDGES=1 ile anlamli"
EVAL_TEST = os.environ.get("EVAL_TEST", "1") == "1"
STRICT_SPLIT = os.environ.get("STRICT_SPLIT", "0") == "1"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "100"))
KOL = "SP_GAT"
SP_N_SEGMENTS = 1024
SP_COMPACTNESS = 0.1
IZ_BLOK = 4
PARMAK_IZI_N = 64
assert not MOTION_EDGES and HORIZON_GAT, "superpiksel kollari: MOTION_EDGES=0, HORIZON_GAT=1 (ufuk islemi kola gore)"

# settings

SEED      = int(os.environ.get("SEED", "42"))
MODEL_TAG = "sp_gat_lagrange_res" + os.environ.get("TAG_SUFFIX", "")

DATA_DIR  = os.environ.get("DATA_DIR", "./data")
OUT_DIR   = os.environ.get("OUT_DIR",  "./outputs")
FIG_DIR   = os.environ.get(
    "FIG_DIR",
    os.path.join("./figures", f"{MODEL_TAG}_seed{SEED}"))

TRAIN_FILE = os.path.join(DATA_DIR, "meteonet_nw_train.npy")
VAL_FILE   = os.path.join(DATA_DIR, "meteonet_nw_val.npy")
TEST_FILE  = os.path.join(DATA_DIR, "meteonet_nw_test.npy")
FLOW_DIR   = os.environ.get("FLOW_DIR", DATA_DIR)
FLOW_METHOD = os.environ.get("FLOW_METHOD", "tvl1")
assert FLOW_METHOD in ("ilk", "tvl1"), FLOW_METHOD
def flow_file(data_file):
    stem = os.path.basename(data_file)[:-len(".npy")]
    return os.path.join(FLOW_DIR, f"{stem}_akis56_{FLOW_METHOD}.npy")

MODEL_SAVE_PATH = os.path.join(OUT_DIR, f"{MODEL_TAG}_seed{SEED}.pt")
METRICS_PATH    = os.path.join(OUT_DIR, f"{MODEL_TAG}_seed{SEED}_metrics.json")

N_INPUT    = 7
N_PREDICT  = 6
IMG_SIZE   = 128
GRID_SIZE  = 32

FEAT_DIM   = 64
HIDDEN_DIM = 64
GAT_HEADS  = 4
N_LAYERS   = 2
SIGMA      = 1.0

EPOCHS     = 1 if SMOKE_TEST else MAX_EPOCHS
PATIENCE   = 15
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
LR         = 1e-3
BATCH_SIZE = 4

# thresholds 0.1, 0.5, 1, 2, 5, 10 mm/h divided by the 12 mm/h cap
CSI_THRESHOLDS = [0.1/12, 0.5/12, 1.0/12, 2.0/12, 5.0/12, 10.0/12]

# weight bins (mm/h): <0.1, 0.1-0.5, 0.5-1, 1-2, 2-5, 5-10, >=10
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
      f"  |  MODE: {PREDICTION_MODE} | HORIZON_GAT: {int(HORIZON_GAT)}")
print(f"Çıktı: {MODEL_SAVE_PATH}")

if not SELF_TEST:
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
        self.flow = np.load(flow_file(filepath), mmap_mode='r')
        assert self.flow.shape == (self.data.shape[0], 2, IMG_SIZE, IMG_SIZE), (
            f"akis dosyasi uyusmuyor: {self.flow.shape} vs {self.data.shape}")
        self.flow_manifest, self.flow_kimlik = AM.dogrula(filepath, flow_file(filepath), FLOW_METHOD,
                                                          self.data.shape[0])
        self.veri_kimlik = {"dogrulanmis_kaynak_sha256": self.flow_kimlik["dogrulanmis_kaynak_sha256"],
                            "kaynak": "akis_tam_dogrulamasi", "bu_kosuda_hash": self.flow_kimlik["bu_kosuda_hash"]}
        self.idx = None
        if STRICT_SPLIT:
            self.idx = VS.secim(VS.split_adi(filepath), self.data.shape[0],
                                kaynak_sha256=self.flow_kimlik["dogrulanmis_kaynak_sha256"])
            if max_samples is not None:
                self.idx = self.idx[:max_samples]
            self.n = int(self.idx.size)
        else:
            if max_samples is not None:
                self.data = self.data[:max_samples]
                self.flow = self.flow[:max_samples]
            self.n = self.data.shape[0]
        self.orijinal_indeksler = self.idx if self.idx is not None else np.arange(self.n)
        print(f"  {self.n} örnek  {self.data.shape}  akis {self.flow.shape}  STRICT_SPLIT={int(STRICT_SPLIT)}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.idx is not None:
            idx = int(self.idx[idx])
        seq = self.data[idx].astype(np.float32)
        seq = seq[::2]   # 25 maps at 5 min -> 13 maps at 10 min
        x = torch.from_numpy(seq[:N_INPUT]).unsqueeze(1)
        y = torch.from_numpy(seq[N_INPUT:]).unsqueeze(1)
        v = np.asarray(self.flow[idx], dtype=np.float32)
        base = np.stack([LK.advect(seq[N_INPUT - 1], v, k) for k in range(1, N_PREDICT + 1)])
        b = torch.from_numpy(base.astype(np.float32)).unsqueeze(1)
        # superpixels are built from the last input map only
        return x, y, b, torch.from_numpy(etiket_haritasi(seq[N_INPUT - 1]))

# graph helpers

def build_grid_edges(H=GRID_SIZE, W=GRID_SIZE):
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        edges.append([node, ni * W + nj])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def batch_edge_index(edge_index_single, batch_size, num_nodes):
    return torch.cat([edge_index_single + i * num_nodes
                      for i in range(batch_size)], dim=1)

def advective_edges(v32, k, G):
    B = v32.shape[0]
    dev = v32.device
    rr, cc = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij")
    rr = rr.unsqueeze(0).expand(B, G, G).float()
    cc = cc.unsqueeze(0).expand(B, G, G).float()
    pr = rr + k * v32[:, 0]
    pc = cc + k * v32[:, 1]
    r0, c0 = torch.floor(pr), torch.floor(pc)
    fr, fc = pr - r0, pc - c0
    bidx = torch.arange(B, device=dev).view(B, 1, 1).expand(B, G, G)
    dst = (bidx * G * G + rr.long() * G + cc.long())
    srcs, dsts, ws = [], [], []
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rs, cs = r0 + dr, c0 + dc
        w = (fr if dr else 1 - fr) * (fc if dc else 1 - fc)
        ok = (rs >= 0) & (rs < G) & (cs >= 0) & (cs < G) & (w > 1e-6)
        src = bidx * G * G + rs.long().clamp(0, G - 1) * G + cs.long().clamp(0, G - 1)
        srcs.append(src[ok]); dsts.append(dst[ok]); ws.append(w[ok])
    ei = torch.stack([torch.cat(srcs), torch.cat(dsts)])
    return ei, torch.cat(ws).unsqueeze(1)

def union_motion_edges(edge_index, ei_a, n_total):
    ei_a = ei_a[:, ei_a[0] != ei_a[1]]
    key8 = edge_index[0] * n_total + edge_index[1]
    key_a = torch.unique(ei_a[0] * n_total + ei_a[1])
    key_a = key_a[~torch.isin(key_a, key8)]
    new = torch.stack([key_a // n_total, key_a % n_total])
    return torch.cat([edge_index, new], dim=1), new.shape[1]

def compute_edge_attr(x_nodes, edge_index, sigma=SIGMA):
    src, dst = edge_index[0], edge_index[1]
    diff    = x_nodes[src] - x_nodes[dst]
    dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

# encoder and decoder

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=FEAT_DIM):
        super().__init__()
        mid = out_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(mid, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class CNNDecoder(nn.Module):
    def __init__(self, in_channels=HIDDEN_DIM + FEAT_DIM, out_channels=1):
        super().__init__()
        mid = HIDDEN_DIM // 2
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid, 2, stride=2),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(mid, mid // 2, 2, stride=2),
            nn.GroupNorm(4, mid // 2), nn.LeakyReLU(0.2),
            nn.Conv2d(mid // 2, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# GAT-LSTM cell

class GraphConvLSTMCell(nn.Module):
    def __init__(self, in_feat, hidden_dim=HIDDEN_DIM, heads=GAT_HEADS):
        super().__init__()
        assert in_feat == hidden_dim, (
            f"Residual connection h_next + x requires in_feat == hidden_dim, "
            f"got in_feat={in_feat}, hidden_dim={hidden_dim}"
        )
        assert (4 * hidden_dim) % heads == 0, (
            f"4*hidden_dim ({4 * hidden_dim}) must be divisible by heads ({heads})"
        )
        self.hidden_dim = hidden_dim
        self.gat = GATConv(
            in_channels   = in_feat + hidden_dim,
            out_channels  = (4 * hidden_dim) // heads,
            heads         = heads,
            concat        = True,
            edge_dim      = 1,
            add_self_loops= True,
            fill_value    = 1.0,   # self-loop edge weight = 1
        )

        # forget-gate bias = 1; gate order i, f, g, o
        with torch.no_grad():
            self.gat.bias[hidden_dim:2 * hidden_dim].fill_(1.0)

    def forward(self, x, h, c, edge_index, edge_attr):
        gates = self.gat(torch.cat([x, h], dim=-1), edge_index,
                         edge_attr=edge_attr)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + x   # residual
        return h_next, c_next


class MeteoNetGATLagrangeResidual(nn.Module):
    def __init__(self, in_channels=1, feat_dim=FEAT_DIM,
                 hidden_dim=HIDDEN_DIM, heads=GAT_HEADS,
                 n_layers=N_LAYERS, grid_size=GRID_SIZE,
                 n_predict=N_PREDICT):
        super().__init__()
        if feat_dim != hidden_dim:
            raise ValueError("Mevcut residual GAT-LSTM hucreleri feat_dim == hidden_dim gerektirir.")
        if hidden_dim % heads:
            raise ValueError("hidden_dim, GAT_HEADS sayisina bolunebilmelidir.")
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_predict = n_predict
        self.use_horizon_gat = HORIZON_GAT
        self.motion_edges = MOTION_EDGES
        self.encoder = CNNEncoder(in_channels, feat_dim)
        self.decoder = CNNDecoder(hidden_dim + feat_dim, in_channels)
        self.cells = nn.ModuleList([
            GraphConvLSTMCell(in_feat=feat_dim if i == 0 else hidden_dim,
                              hidden_dim=hidden_dim, heads=heads)
            for i in range(n_layers)
        ])
        self.register_buffer("edge_index_single", build_grid_edges(grid_size, grid_size))

        self.horizon_embedding = nn.Embedding(n_predict, hidden_dim)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        self.horizon_proj = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_gat = (
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                edge_dim=1,
                add_self_loops=True,
                fill_value=1.0,
            ) if self.use_horizon_gat else None
        )
        self.horizon_norm = nn.LayerNorm(hidden_dim)
        # zero-initialised last layer: the untrained model outputs Lagrangian persistence
        nn.init.zeros_(self.decoder.net[-1].weight)
        nn.init.zeros_(self.decoder.net[-1].bias)

    def _encode(self, x_in, edge_index):
        B, T, _, H, W = x_in.shape
        N = self.num_nodes
        h_list = [x_in.new_zeros((B * N, self.hidden_dim))
                  for _ in range(self.n_layers)]
        c_list = [x_in.new_zeros((B * N, self.hidden_dim))
                  for _ in range(self.n_layers)]
        last_feat = None
        last_edge_attr = None

        for t in range(T):
            feat = self.encoder(x_in[:, t])
            nodes = feat.flatten(2).transpose(1, 2).reshape(B * N, self.feat_dim)
            edge_attr = compute_edge_attr(nodes, edge_index)
            new_h, new_c = [], []
            z = nodes
            for k, cell in enumerate(self.cells):
                hk, ck = cell(z, h_list[k], c_list[k], edge_index, edge_attr)
                new_h.append(hk)
                new_c.append(ck)
                z = hk
            h_list, c_list = new_h, new_c
            last_feat = feat
            last_edge_attr = edge_attr
        return h_list[-1], last_feat, last_edge_attr

    def _decode_horizon(self, encoded_h, base_feat, edge_index, edge_attr, lead_idx, B):
        N = self.num_nodes
        time_bias = self.horizon_proj(self.horizon_embedding.weight[lead_idx])
        time_conditioned = encoded_h + time_bias.unsqueeze(0)

        if self.use_horizon_gat:
            graph_out = self.horizon_gat(time_conditioned, edge_index,
                                         edge_attr=edge_attr)
            graph_out = self.horizon_norm(time_conditioned + graph_out)
        else:
            graph_out = self.horizon_norm(time_conditioned)
        graph_out = F.leaky_relu(graph_out, negative_slope=0.2)
        fmap = graph_out.reshape(B, N, self.hidden_dim).transpose(1, 2) \
                         .reshape(B, self.hidden_dim, self.grid_size, self.grid_size)
        return self.decoder(torch.cat([fmap, base_feat], dim=1))

    def forward(self, x_in, base, v32=None):
        if x_in.ndim != 5:
            raise ValueError(f"Girdi (B,T,C,H,W) olmali, gelen: {tuple(x_in.shape)}")
        B, T, C, H, W = x_in.shape
        if T != N_INPUT or H != 4 * self.grid_size or W != 4 * self.grid_size:
            raise ValueError(
                f"Beklenen T={N_INPUT}, H=W={4*self.grid_size}; gelen: {tuple(x_in.shape)}"
            )
        edge_index = batch_edge_index(
            self.edge_index_single, B, self.num_nodes).to(x_in.device)
        if base.shape != (B, self.n_predict, C, H, W):
            raise ValueError(f"base (B,{self.n_predict},C,H,W) olmali, gelen: {tuple(base.shape)}")
        encoded_h, last_feat, edge_attr = self._encode(x_in, edge_index)
        if self.motion_edges:
            if v32 is None or v32.shape != (B, 2, self.grid_size, self.grid_size):
                raise ValueError("MOTION_EDGES=1: v32 (B,2,G,G) gerekli")
            nodes_last = last_feat.flatten(2).transpose(1, 2).reshape(B * self.num_nodes, self.feat_dim)
            v_e = v32 if MOTION_SIGN == 1 else -v32
            outputs = []
            for k in range(self.n_predict):
                ei_a, _ = advective_edges(v_e, k + 1, self.grid_size)
                ei_k, _ = union_motion_edges(edge_index, ei_a, B * self.num_nodes)
                ea_k = compute_edge_attr(nodes_last, ei_k)
                outputs.append((base[:, k] + self._decode_horizon(
                    encoded_h, self.encoder(base[:, k]), ei_k, ea_k, k, B)).unsqueeze(1))
        else:
            outputs = [
                (base[:, k] + self._decode_horizon(encoded_h, self.encoder(base[:, k]), edge_index,
                                                   edge_attr, k, B)).unsqueeze(1)
                for k in range(self.n_predict)
            ]
        return torch.cat(outputs, dim=1)


# superpixel graph model

def etiket_haritasi(kare, kol=None, n_segments=None):
    kol = KOL if kol is None else kol
    n_segments = SP_N_SEGMENTS if n_segments is None else n_segments
    kare = np.asarray(kare, dtype=np.float64)
    H, W = kare.shape
    if kol == "IZ_GAT":
        assert H % IZ_BLOK == 0 and W % IZ_BLOK == 0, (H, W, IZ_BLOK)
        ii, jj = np.meshgrid(np.arange(H) // IZ_BLOK, np.arange(W) // IZ_BLOK, indexing="ij")
        lab = ii * (W // IZ_BLOK) + jj
    else:
        lab = slic(kare, n_segments=n_segments, compactness=SP_COMPACTNESS, channel_axis=None,
                   start_label=0, enforce_connectivity=True)
        _, lab = np.unique(lab, return_inverse=True)
        lab = lab.reshape(H, W)
    lab = lab.astype(np.int64)
    assert lab.min() == 0 and np.bincount(lab.ravel()).min() > 0, "bos bolge var"
    return lab


def etiket_parmak_izi(ds, n=None):
    import hashlib as _hl
    n = min(int(PARMAK_IZI_N if n is None else n), len(ds))
    h = _hl.sha256()
    toplam = 0
    for i in range(n):
        e = ds[i][3]
        e = e.numpy() if hasattr(e, "numpy") else np.asarray(e)
        e = np.ascontiguousarray(e, dtype="<i8")
        h.update(np.asarray(e.shape, dtype="<i8").tobytes())
        h.update(e.tobytes())
        toplam += int(e.max()) + 1
    return {"n": int(n), "sha256": h.hexdigest(), "bolge_sayisi_toplam": int(toplam)}


def bolge_kenarlari(eg, N):
    a = torch.cat([eg[:, :, :-1].reshape(-1), eg[:, :-1, :].reshape(-1)])
    b = torch.cat([eg[:, :, 1:].reshape(-1), eg[:, 1:, :].reshape(-1)])
    m = a != b
    lo, hi = torch.minimum(a[m], b[m]), torch.maximum(a[m], b[m])
    key = torch.unique(lo * N + hi)
    lo, hi = key // N, key % N
    return torch.cat([torch.stack([lo, hi]), torch.stack([hi, lo])], dim=1).contiguous()


class CNNEncoderTam(nn.Module):
    def __init__(self, in_channels=1, out_channels=FEAT_DIM):
        super().__init__()
        mid = out_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, stride=1, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, stride=1, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class CNNDecoderTam(nn.Module):
    def __init__(self, in_channels=HIDDEN_DIM + FEAT_DIM, out_channels=1):
        super().__init__()
        mid = HIDDEN_DIM // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, stride=1, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, stride=1, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid // 2, 3, stride=1, padding=1),
            nn.GroupNorm(4, mid // 2), nn.LeakyReLU(0.2),
            nn.Conv2d(mid // 2, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class NodeLSTMCell(nn.Module):
    def __init__(self, in_feat, hidden_dim=HIDDEN_DIM):
        super().__init__()
        assert in_feat == hidden_dim, (in_feat, hidden_dim)
        self.hidden_dim = hidden_dim
        self.lin = nn.Linear(in_feat + hidden_dim, 4 * hidden_dim)
        with torch.no_grad():
            self.lin.bias[hidden_dim:2 * hidden_dim].fill_(1.0)   # forget-gate bias = 1

    def forward(self, x, h, c, edge_index=None, edge_attr=None):
        gates = self.lin(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + x   # residual
        return h_next, c_next


class MeteoNetSPLagrange(nn.Module):
    def __init__(self, kol=None, in_channels=1, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, heads=GAT_HEADS,
                 n_layers=N_LAYERS, n_predict=N_PREDICT):
        super().__init__()
        self.kol = KOL if kol is None else kol
        assert self.kol in ("SP_GAT", "IZ_GAT", "SP_BAGIMSIZ"), self.kol
        assert feat_dim == hidden_dim and (4 * hidden_dim) % heads == 0 and hidden_dim % heads == 0
        self.mesaj = self.kol != "SP_BAGIMSIZ"
        self.feat_dim, self.hidden_dim, self.n_layers, self.n_predict = feat_dim, hidden_dim, n_layers, n_predict
        self.encoder = CNNEncoderTam(in_channels, feat_dim)
        self.decoder = CNNDecoderTam(hidden_dim + feat_dim, in_channels)
        if self.mesaj:
            self.cells = nn.ModuleList([GraphConvLSTMCell(in_feat=hidden_dim, hidden_dim=hidden_dim, heads=heads)
                                        for _ in range(n_layers)])
            self.horizon_op = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads,
                                      concat=True, edge_dim=1, add_self_loops=True, fill_value=1.0)
        else:
            self.cells = nn.ModuleList([NodeLSTMCell(in_feat=hidden_dim, hidden_dim=hidden_dim)
                                        for _ in range(n_layers)])
            self.horizon_op = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_embedding = nn.Embedding(n_predict, hidden_dim)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        self.horizon_proj = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_norm = nn.LayerNorm(hidden_dim)
        # zero-initialised last layer: the untrained model outputs Lagrangian persistence
        nn.init.zeros_(self.decoder.net[-1].weight)
        nn.init.zeros_(self.decoder.net[-1].bias)

    @staticmethod
    def kuresel_etiket(etiket):
        B = etiket.shape[0]
        n_b = etiket.reshape(B, -1).max(dim=1).values + 1
        ofs = torch.cumsum(n_b, 0) - n_b
        return etiket + ofs.view(B, 1, 1), int(n_b.sum()), n_b

    @staticmethod
    def havuzla(feat, eg, N):
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(-1, C).float()
        idx = eg.reshape(-1)
        top = torch.zeros(N, C, device=feat.device, dtype=torch.float32).index_add_(0, idx, flat)
        say = torch.bincount(idx, minlength=N).clamp(min=1).to(torch.float32)
        return top / say.unsqueeze(1)

    @staticmethod
    def yay(dugum, eg):
        B, H, W = eg.shape
        return dugum[eg.reshape(-1)].reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def hucreler(self, dugum_dizisi, edge_index, h0=None, c0=None):
        N = dugum_dizisi[0].shape[0]
        dev = dugum_dizisi[0].device
        h = [torch.zeros(N, self.hidden_dim, device=dev) if h0 is None else h0[k] for k in range(self.n_layers)]
        c = [torch.zeros(N, self.hidden_dim, device=dev) if c0 is None else c0[k] for k in range(self.n_layers)]
        iz, ea = [], None
        for nodes in dugum_dizisi:
            ea = compute_edge_attr(nodes, edge_index) if self.mesaj else None
            z, nh, nc = nodes, [], []
            for k, cell in enumerate(self.cells):
                hk, ck = cell(z, h[k], c[k], edge_index, ea)
                nh.append(hk); nc.append(ck); z = hk
            h, c = nh, nc
            iz.append((h, c))
        return h[-1], iz, ea

    def ufuk_dugum(self, encoded_h, edge_index, edge_attr, k):
        tc = encoded_h + self.horizon_proj(self.horizon_embedding.weight[k]).unsqueeze(0)
        if self.mesaj:
            out = self.horizon_op(tc, edge_index, edge_attr=edge_attr)
        else:
            out = self.horizon_op(tc)
        return F.leaky_relu(self.horizon_norm(tc + out), negative_slope=0.2)

    def ara(self, x_in, base, etiket):
        B, T, C, H, W = x_in.shape
        eg, N, _ = self.kuresel_etiket(etiket)
        ei = bolge_kenarlari(eg, N)
        dizi = [self.havuzla(self.encoder(x_in[:, t]), eg, N) for t in range(T)]
        enc_h, _, ea = self.hucreler(dizi, ei)
        ud = [self.ufuk_dugum(enc_h, ei, ea, k) for k in range(self.n_predict)]
        return eg, N, ei, enc_h, ud, [self.yay(u, eg) for u in ud]

    def forward(self, x_in, base, etiket):
        if x_in.ndim != 5 or x_in.shape[1] != N_INPUT:
            raise ValueError(f"Girdi (B,{N_INPUT},C,H,W) olmali: {tuple(x_in.shape)}")
        B, T, C, H, W = x_in.shape
        if base.shape != (B, self.n_predict, C, H, W) or etiket.shape != (B, H, W):
            raise ValueError(f"base/etiket sekli: {tuple(base.shape)} {tuple(etiket.shape)}")
        etiket = etiket.to(x_in.device).long()
        _, _, _, _, _, rasterler = self.ara(x_in, base, etiket)
        out = [(base[:, k] + self.decoder(torch.cat([rasterler[k], self.encoder(base[:, k])], dim=1))).unsqueeze(1)
               for k in range(self.n_predict)]
        return torch.cat(out, dim=1)

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
    # max-pool, then threshold
    t = torch.from_numpy(arr).float().unsqueeze(1)               # (N, 1, H, W)
    pooled = F.max_pool2d(t, kernel_size=pool_size, stride=pool_size)
    return pooled.squeeze(1).numpy()

def evaluate_metrics(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb, *ex in loader:
            pred = model(xb.to(DEVICE), *[e.to(DEVICE) for e in ex]).cpu().numpy()
            assert np.isfinite(pred).all(), "tahminde NaN/Inf var (kirpmadan once)"
            pred = np.clip(pred, 0.0, 1.0)   # no sigmoid: clip the output
            all_pred.append(pred[:, :, 0])
            all_true.append(yb.numpy()[:, :, 0])
    pred_all = np.concatenate(all_pred)                 # (N, T_pred, H, W)
    true_all = np.concatenate(all_true)
    N, T_pred, H, W = pred_all.shape

    ssim_matrix = np.zeros((N, T_pred), dtype=np.float64)
    for i in range(N):
        for t in range(T_pred):
            ssim_matrix[i, t] = sk_ssim(
                pred_all[i, t], true_all[i, t], data_range=1.0)

    per_lt_mse, per_lt_ssim, per_lt_csi_m, per_lt_csi_per_thr = [], [], [], []
    for t in range(T_pred):
        p_t = pred_all[:, t]
        y_t = true_all[:, t]
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
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
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
                  "lr": [], "grad_norm": [], "grad_norm_inf": [], "amp_scale": [],
                  "n_steps_finite_gradnorm": [], "ckpt_saved_epochs": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = __import__('time').time()
        model.train()
        train_loss = 0.0; n_batches = 0; n_train_seen = 0
        gn_sum = 0.0; gn_n = 0; gn_inf = 0
        for xb, yb, *ex in train_loader:
            xb, yb, ex = xb.to(DEVICE), yb.to(DEVICE), [e.to(DEVICE) for e in ex]
            if global_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg['lr'] = LR * (global_step + 1) / warmup_steps
            global_step += 1
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
                pred = model(xb, *ex)
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
            for xb, yb, *ex in val_loader:
                xb, yb, ex = xb.to(DEVICE), yb.to(DEVICE), [e.to(DEVICE) for e in ex]
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
                    val_loss += criterion(model(xb, *ex), yb).item() * xb.size(0)
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
        history["n_steps_finite_gradnorm"].append(int(gn_n))
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
            history["ckpt_saved_epochs"].append(epoch)
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
    path = os.path.join(FIG_DIR, "meteonet_gat_lagrange_res_comparison.gif")
    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF kaydedildi: {path}")

# main

def _seed_worker(worker_id):
    ws = SEED + worker_id
    np.random.seed(ws)
    random.seed(ws)


if __name__ == "__main__":
    if SELF_TEST:
        import copy
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        Hs = 32
        NSEG = 48

        def _kare():
            yy, xx = np.mgrid[0:Hs, 0:Hs]
            z = np.zeros((Hs, Hs))
            for _ in range(3):
                cy, cx, s, a = rng.uniform(4, 28), rng.uniform(4, 28), rng.uniform(2, 6), rng.uniform(0.2, 1.0)
                z += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
            return np.clip(np.round(z, 2), 0, 1)

        def _veri(B):
            x = torch.from_numpy(np.stack([[_kare() for _ in range(N_INPUT)] for _ in range(B)])).float().unsqueeze(2)
            base = torch.from_numpy(np.stack([[_kare() for _ in range(N_PREDICT)] for _ in range(B)])).float().unsqueeze(2)
            et = torch.from_numpy(np.stack([etiket_haritasi(x[b, -1, 0].numpy(), n_segments=NSEG) for b in range(B)]))
            return x, base, et

        x2, b2, e2 = _veri(2)
        m = MeteoNetSPLagrange().cpu()
        m.eval()
        with torch.no_grad():
            assert torch.equal(m(x2, b2, e2), b2), "Baslangicta cikti Lagrange tabanina esit degil"
        mt = copy.deepcopy(m).cpu()
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            mt.decoder.net[-1].weight.copy_(torch.randn(mt.decoder.net[-1].weight.shape, generator=g) * 0.1)
            mt.decoder.net[-1].bias.fill_(0.01)
        with torch.no_grad():
            assert not torch.equal(mt(x2, b2, e2), b2), "sinama basligi sifir-olmayan degil"
        with torch.no_grad():
            eg, N, ei, eh, ud, rs = mt.ara(x2, b2, e2)
            n0 = int(e2[0].max()) + 1
            ornek = (torch.arange(N) >= n0).long()
            assert torch.equal(ornek[ei[0]], ornek[ei[1]]), "ornekler arasi kenar var"
            assert not (ei[0] == ei[1]).any(), "edge_index oz-dongu iceriyor (GATConv ekler)"
            assert eh.shape == (N, HIDDEN_DIM) and rs[0].shape == (2, HIDDEN_DIM, Hs, Hs)
            for bi in range(2):
                _, Nb, _, ehb, udb, rsb = mt.ara(x2[bi:bi + 1], b2[bi:bi + 1], e2[bi:bi + 1])
                sl = slice(0, n0) if bi == 0 else slice(n0, N)
                assert torch.allclose(eh[sl], ehb, atol=1e-5), f"yigin/tek ornek encoded_h farki (ornek {bi})"
                for k in range(N_PREDICT):
                    assert torch.allclose(rs[k][bi:bi + 1], rsb[k], atol=1e-5), f"yigin/tek ornek raster farki k={k}"
                assert torch.allclose(mt(x2, b2, e2)[bi:bi + 1], mt(x2[bi:bi + 1], b2[bi:bi + 1], e2[bi:bi + 1]),
                                      atol=1e-5), f"yigin/tek ornek cikti farki (ornek {bi})"
        with torch.no_grad():
            p = torch.randperm(n0, generator=g)
            e2p = e2.clone()
            e2p[0] = p[e2[0]]
            _, _, _, ehp, udp, rsp = mt.ara(x2, b2, e2p)
            assert torch.allclose(ehp[:n0][p], eh[:n0], atol=1e-5), "permutasyonda dugum durumu tasinmadi"
            for k in range(N_PREDICT):
                assert torch.allclose(rsp[k], rs[k], atol=1e-5), f"permutasyonda raster degisti k={k}"
            assert torch.allclose(mt(x2, b2, e2p), mt(x2, b2, e2), atol=1e-5), "permutasyonda cikti degisti"
        for bi in range(2):
            assert torch.bincount(e2[bi].reshape(-1)).min() > 0, "bos bolge"
        with torch.no_grad():
            eg, N, ei, _, _, _ = mt.ara(x2, b2, e2)
            dizi = [mt.havuzla(mt.encoder(x2[:, t]), eg, N) for t in range(N_INPUT)]
            j = 3
            diger = torch.arange(N) != j
            _, iz0, ea0 = mt.hucreler(dizi, ei)
            dizi_b = [d.clone() for d in dizi]
            for d in dizi_b:
                d[j] += 0.5
            _, iz1, _ = mt.hucreler(dizi_b, ei)
            h0 = [torch.zeros(N, HIDDEN_DIM) for _ in range(N_LAYERS)]
            c0 = [torch.zeros(N, HIDDEN_DIM) for _ in range(N_LAYERS)]
            h0b = [h.clone() for h in h0]; c0b = [c.clone() for c in c0]
            h0b[0][j] += 0.5; c0b[1][j] -= 0.5
            _, iz2, _ = mt.hucreler(dizi, ei, h0b, c0b)
            fark_g = max(float((a[0][L][diger] - b[0][L][diger]).abs().max()) for a, b in zip(iz0, iz1) for L in range(N_LAYERS))
            fark_g = max(fark_g, max(float((a[1][L][diger] - b[1][L][diger]).abs().max()) for a, b in zip(iz0, iz1) for L in range(N_LAYERS)))
            fark_hc = max(float((a[s][L][diger] - b[s][L][diger]).abs().max())
                          for a, b in zip(iz0, iz2) for L in range(N_LAYERS) for s in (0, 1))
            hb = iz0[-1][0][-1].clone()
            hb2 = hb.clone(); hb2[j] += 0.5
            fark_u = max(float((mt.ufuk_dugum(hb, ei, ea0, k)[diger] - mt.ufuk_dugum(hb2, ei, ea0, k)[diger]).abs().max())
                         for k in range(N_PREDICT))
            print(f"OLCUM_MESAJ kol={KOL} girdi_bozma_diger_max={fark_g:.3e} hc_bozma_diger_max={fark_hc:.3e} "
                  f"ufuk_bozma_diger_max={fark_u:.3e}")
            if KOL == "SP_BAGIMSIZ":
                assert fark_g == 0.0 and fark_hc == 0.0 and fark_u == 0.0, "SP_BAGIMSIZ'da dugumler arasi bilgi akisi var"
            else:
                assert fark_g > 0 and fark_u > 0, "GAT kolunda dugumler arasi bilgi akisi gorulmedi (sinama duyarsiz)"
        _iz = torch.from_numpy(etiket_haritasi(np.zeros((IMG_SIZE, IMG_SIZE)), kol="IZ_GAT")).unsqueeze(0)
        _eg, _N, _ = MeteoNetSPLagrange.kuresel_etiket(_iz)
        _ei = bolge_kenarlari(_eg, _N)
        print(f"OLCUM_IZ bolge={_N} yonlu_kenar={_ei.shape[1]}")
        assert _N == 1024 and _ei.shape[1] == 3968, (_N, _ei.shape)
        assert torch.equal(torch.from_numpy(etiket_haritasi(x2[0, -1, 0].numpy(), n_segments=NSEG)), e2[0])
        _sahte = [(None, None, None, torch.from_numpy(etiket_haritasi(x2[b, -1, 0].numpy(), n_segments=NSEG)))
                  for b in range(x2.shape[0])]
        _p1, _p2 = etiket_parmak_izi(_sahte, n=2), etiket_parmak_izi(_sahte, n=2)
        assert _p1 == _p2 and _p1["n"] == 2, (_p1, _p2)
        assert etiket_parmak_izi(_sahte, n=1)["sha256"] != _p1["sha256"], "parmak izi n'ye duyarsiz"
        _boz = [list(t) for t in _sahte]; _boz[1][3] = _boz[1][3].clone(); _boz[1][3][0, 0] += 1
        assert etiket_parmak_izi(_boz, n=2)["sha256"] != _p1["sha256"], "parmak izi tek etikete duyarsiz"
        assert etiket_parmak_izi(_sahte, n=99)["n"] == len(_sahte), "n uzunlukla sinirlanmiyor"
        print(f"OLCUM_PARMAK_IZI n=2 sha256={_p1['sha256'][:16]} bolge_toplam={_p1['bolge_sayisi_toplam']}")
        _e1 = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
        _eg, _N, _ = MeteoNetSPLagrange.kuresel_etiket(torch.stack([_e1, _e1]))
        assert _N == 8 and torch.equal(_eg[1], _e1 + 4), "ofset hatali"
        _ei = bolge_kenarlari(_eg, _N)
        _bek = {(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)}
        _bek = _bek | {(a + 4, b + 4) for a, b in _bek}
        assert set(map(tuple, _ei.t().tolist())) == _bek and _ei.shape[1] == 16, f"elle kenar: {_ei.t().tolist()}"
        _f = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
        _d = MeteoNetSPLagrange.havuzla(_f, _eg, _N)
        assert torch.equal(_d[:, 0], torch.tensor([2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5])), f"elle havuz: {_d[:, 0]}"
        _r = MeteoNetSPLagrange.yay(_d, _eg)
        assert torch.equal(_r[0, 0], torch.tensor([[2.5, 2.5, 4.5, 4.5], [2.5, 2.5, 4.5, 4.5],
                                                   [10.5, 10.5, 12.5, 12.5], [10.5, 10.5, 12.5, 12.5]])), "elle yayma"
        m.train()
        tgt = torch.rand_like(b2)
        loss = bmse_loss(m(x2, b2, e2), tgt)
        assert torch.isfinite(loss)
        loss.backward()
        assert m.decoder.net[-1].weight.grad is not None and m.decoder.net[-1].weight.grad.abs().sum() > 0
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        m.zero_grad()
        bmse_loss(m(x2, b2, e2), tgt).backward()
        for _ad, _mod in (("encoder", m.encoder), ("cells[0]", m.cells[0]), ("horizon_op", m.horizon_op)):
            _gs = sum(float(q.grad.abs().sum()) for q in _mod.parameters() if q.grad is not None)
            assert _gs > 0, f"{_ad} bir guncellemeden sonra da gradyan almiyor"
        for _ad, q in m.named_parameters():
            assert q.grad is None or torch.isfinite(q.grad).all(), f"sonlu olmayan gradyan: {_ad}"
        n_par = sum(q.numel() for q in m.parameters() if q.requires_grad)
        if DEVICE.type == "cuda":
            import time as _time, gc as _gc

            def _sinama_basligi(mm):
                _g = torch.Generator().manual_seed(11)
                with torch.no_grad():
                    mm.decoder.net[-1].weight.copy_(torch.randn(mm.decoder.net[-1].weight.shape, generator=_g) * 0.1)
                    mm.decoder.net[-1].bias.fill_(0.01)
                return mm

            def _amp_adim(mm, xg, bg, egx, ad):
                mm.zero_grad()
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    og = mm(xg, bg, egx)
                    lg = bmse_loss(og, torch.rand_like(og))
                assert torch.isfinite(og).all() and torch.isfinite(lg), f"CUDA/AMP cikti/kayip sonlu degil ({ad})"
                lg.backward()
                for _ad, q in mm.named_parameters():
                    assert q.grad is None or torch.isfinite(q.grad).all(), f"CUDA/AMP sonlu olmayan gradyan ({ad}): {_ad}"
                for _ad, _mod in (("encoder", mm.encoder), ("cells[0]", mm.cells[0]), ("horizon_op", mm.horizon_op),
                                  ("decoder", mm.decoder)):
                    _gs = sum(float(q.grad.abs().sum()) for q in _mod.parameters() if q.grad is not None)
                    assert _gs > 0, f"CUDA/AMP {ad}: {_ad} gradyan almiyor"
                return float(lg)

            def _kucuk():
                mm = _sinama_basligi(copy.deepcopy(m)).to(DEVICE)
                return _amp_adim(mm, x2.to(DEVICE), b2.to(DEVICE), e2.to(DEVICE), "32x32_B2")

            def _tam():
                _rngf = np.random.default_rng(1)
                xf = torch.from_numpy(_rngf.random((BATCH_SIZE, N_INPUT, 1, IMG_SIZE, IMG_SIZE)) ** 8).float()
                bf = torch.from_numpy(_rngf.random((BATCH_SIZE, N_PREDICT, 1, IMG_SIZE, IMG_SIZE)) ** 8).float()
                ef = torch.from_numpy(np.stack([etiket_haritasi(xf[b, -1, 0].numpy()) for b in range(BATCH_SIZE)]))
                mm = _sinama_basligi(MeteoNetSPLagrange()).to(DEVICE)
                xf, bf, ef = xf.to(DEVICE), bf.to(DEVICE), ef.to(DEVICE)
                torch.cuda.synchronize()
                bas = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats(); _t0 = _time.time()
                lg = _amp_adim(mm, xf, bf, ef, f"tam_B{BATCH_SIZE}_{IMG_SIZE}")
                torch.cuda.synchronize()
                print(f"OLCUM_CUDA_TAM kol={KOL} B={BATCH_SIZE} {IMG_SIZE}x{IMG_SIZE} amp={int(USE_AMP)} "
                      f"dugum={[int(e.max()) + 1 for e in ef]} ileri+geri_s={_time.time() - _t0:.3f} "
                      f"baslangic_allocated_MiB={bas[0] / 2**20:.1f} baslangic_reserved_MiB={bas[1] / 2**20:.1f} "
                      f"tepe_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f} "
                      f"tepe_reserved_MiB={torch.cuda.max_memory_reserved() / 2**20:.1f} kayip={lg:.6f}")

            _kucuk()
            _gc.collect(); torch.cuda.empty_cache()
            _tam()
        print(f"SELF_TEST OK: kol={KOL} param={n_par:,} BMSE={loss.item():.6f} dugum(2 ornek)={N}")
        raise SystemExit(0)

    assert STRICT_SPLIT and not EVAL_TEST and BMSE_WEIGHTS == [1.0] * 7, (
        f"superpiksel hatti STRICT_SPLIT=1 EVAL_TEST=0 BMSE_WEIGHTS=1x7 ister; gelen {int(STRICT_SPLIT)} "
        f"{int(EVAL_TEST)} {BMSE_WEIGHTS}")
    assert os.path.exists(TRAIN_FILE), f"Veri bulunamadı: {TRAIN_FILE}"
    assert os.path.exists(VAL_FILE),   f"Veri bulunamadı: {VAL_FILE}"
    if not SMOKE_TEST and EVAL_TEST:
        assert os.path.exists(TEST_FILE), f"Veri bulunamadı: {TEST_FILE}"

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(SEED)

    max_samples = 16 if SMOKE_TEST else None
    train_ds = MeteoNetDataset(TRAIN_FILE, max_samples=max_samples)
    val_ds   = MeteoNetDataset(VAL_FILE,   max_samples=max_samples)
    SLIC_PARMAK_IZI = etiket_parmak_izi(val_ds)
    print(f"SLIC parmak izi: n={SLIC_PARMAK_IZI['n']} sha256={SLIC_PARMAK_IZI['sha256']} "
          f"bolge_toplam={SLIC_PARMAK_IZI['bolge_sayisi_toplam']}")

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

    # x: (B, 7, 1, 128, 128), y: (B, 6, 1, 128, 128)
    _xb, _yb, _bb, *_ = next(iter(train_loader))
    print(f"Batch shape kontrolü — x: {tuple(_xb.shape)}  y: {tuple(_yb.shape)}  base: {tuple(_bb.shape)}")
    assert _bb.shape == _yb.shape, f"Beklenmedik base shape: {tuple(_bb.shape)}"
    assert _xb.shape[1:] == (N_INPUT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik x shape: {tuple(_xb.shape)} — .npy ham 25-kare mi?")
    assert _yb.shape[1:] == (N_PREDICT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik y shape: {tuple(_yb.shape)}")
    _ilk = next(iter(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)))
    for _i in range(_ilk[0].shape[0]):
        assert np.array_equal(etiket_haritasi(_ilk[0][_i, -1, 0].numpy()), _ilk[3][_i].numpy()), \
            "etiket son gozlenen girdiden gelmiyor"
    print(f"Etiket denetimi: ilk yigin {_ilk[0].shape[0]} dizi, bolge sayilari {[int(e.max()) + 1 for e in _ilk[3]]}")
    del _xb, _yb, _bb, _ilk

    model = MeteoNetSPLagrange(kol=KOL).to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total:,}")
    print(f"Lagrange base ({BASE_TAG}) + GAT-LSTM encoder + horizon GAT residual decoder + B-MSE loss")
    print(f"MOTION_EDGES: {MOTION_EDGES}  MOTION_SIGN: {MOTION_SIGN}  EVAL_TEST: {EVAL_TEST}  NUM_WORKERS: {NUM_WORKERS}")
    print(f"BMSE_WEIGHTS: {BMSE_WEIGHTS}")
    print(f"Sigma: {SIGMA}\n")
    print("Y_hat_k = Warp(Y_t, k*v) + dY_k ; residual head zero-initialised (starts at Lagrange persistence)")
    print(f"HORIZON_GAT enabled: {HORIZON_GAT}")

    history = train_model(model, train_loader, val_loader)
    _sonlu = bool(np.all(np.isfinite(history["train_loss"])) and np.all(np.isfinite(history["val_loss"])))
    _n_adim = int(sum(history["n_steps_finite_gradnorm"]))
    _ckpt = len(history["ckpt_saved_epochs"]) > 0
    print(f"Basari kontrolu: sonlu kayip {_sonlu} | sonlu-gradyan-normlu adim {_n_adim} | bu kosuda checkpoint {_ckpt}")
    if SMOKE_TEST:
        assert _sonlu and _n_adim > 0 and _ckpt, (
            "SMOKE BASARISIZ: sonlu olmayan kayip, sonlu-gradyan-normlu adim yok ya da checkpoint yazilmadi")
    else:
        assert _ckpt, f"Bu kosuda checkpoint yazilmadi (val kaybi hic sonlu/iyilesen olmadi): {MODEL_SAVE_PATH}"

    if SMOKE_TEST:
        print("\n>>> SMOKE TEST BAŞARILI — ortam doğrulandı <<<")
        if DEVICE.type == "cuda":
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    torch_geometric ve ileri/geri yayilim denemesi tamamlandi")
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
            "model":          "MeteoNetSPLagrange",
            "kol":            KOL,
            "sp_n_segments":  SP_N_SEGMENTS,
            "sp_compactness": SP_COMPACTNESS,
            "iz_blok":        IZ_BLOK,
            "skimage":        __import__("skimage").__version__,
            "slic_parmak_izi": SLIC_PARMAK_IZI,
            "model_tag":      MODEL_TAG,
            "operator":       {"SP_GAT": "superpiksel_bolge_GAT_LSTM_ufuk_GAT_residual_on_lagrange",
                               "IZ_GAT": "duzenli_blok_bolge_GAT_LSTM_ufuk_GAT_residual_on_lagrange",
                               "SP_BAGIMSIZ": "superpiksel_bolge_dugum_bagimsiz_LSTM_residual_on_lagrange"}[KOL],
            "base":           BASE_TAG,
            "motion_edges":   MOTION_EDGES,
            "motion_sign":    MOTION_SIGN,
            "eval_test":      EVAL_TEST,
            "strict_split":   STRICT_SPLIT,
            "veri_secimi":    (VS.kimlik() if STRICT_SPLIT else None),
            "n_seq":          {"train": len(train_ds), "val": len(val_ds)},
            "max_epochs":     EPOCHS,
            "flow_dir":       FLOW_DIR,
            "akis_kimligi":   {"train": train_ds.flow_kimlik, "val": val_ds.flow_kimlik},
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
            "rollout":        PREDICTION_MODE,
            "horizon_gat":    HORIZON_GAT,
            "graph_topology": ("yok_dugum_bagimsiz" if KOL == "SP_BAGIMSIZ" else
                               "bolge_sinir_paylasimi_iki_yonlu_gauss_attr"),
            "n_input":       N_INPUT,
            "n_predict":     N_PREDICT,
            "lr_end_of_epoch":     history["lr"],
            "grad_norm_per_epoch": history["grad_norm"],
            "grad_norm_inf_per_epoch": history["grad_norm_inf"],
            "amp_scale_per_epoch": history["amp_scale"],
            "n_steps_finite_gradnorm_per_epoch": history["n_steps_finite_gradnorm"],
            "ckpt_saved_epochs":   history["ckpt_saved_epochs"],
            "all_losses_finite":   _sonlu,
            "ckpt_sha256":         __import__("hashlib").sha256(open(MODEL_SAVE_PATH, "rb").read()).hexdigest(),
            "veri_kimligi":        {sp: dict({"ad": os.path.basename(fp), "boyut_bayt": os.path.getsize(fp)},
                                             **(ds.veri_kimlik or {}))
                                    for sp, fp, ds in (("train", TRAIN_FILE, train_ds), ("val", VAL_FILE, val_ds))},
            "val":  val_metrics,
            "test": test_metrics,
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n--- MeteoNet GAT Lagrange-Residual Sonuclar ---")
        print(f"  Best Val BMSE : {out['best_val_bmse']:.6f}")
        print(f"  Epoch         : {out['epochs_ran']}")
        print(f"  -- Validation (plain MSE on output) --")
        print(f"  MSE        : {val_metrics['mse']:.6f}")
        print(f"  RMSE       : {val_metrics['rmse']:.6f}")
        print(f"  SSIM       : {val_metrics['ssim']:.4f}")
        print(f"  CSI-M      : {val_metrics['csi_m']:.4f}")
        print(f"  CSI-pool4  : {val_metrics['csi_pool4_m']:.4f}")
        print(f"  CSI-pool16 : {val_metrics['csi_pool16_m']:.4f}")
        if test_metrics is not None:
            print(f"  -- Test (held-out, plain MSE on output) --")
            print(f"  MSE        : {test_metrics['mse']:.6f}")
            print(f"  RMSE       : {test_metrics['rmse']:.6f}")
            print(f"  SSIM       : {test_metrics['ssim']:.4f}")
            print(f"  CSI-M      : {test_metrics['csi_m']:.4f}")
            print(f"  CSI-pool4  : {test_metrics['csi_pool4_m']:.4f}")
            print(f"  CSI-pool16 : {test_metrics['csi_pool16_m']:.4f}")
            print("  -- CSI by lead time (10,20,30,40,50,60 min) --")
            print(f"  {test_metrics['per_leadtime']['csi_m']}")
        print("  -- VAL CSI by lead time (10,20,30,40,50,60 min) --")
        print(f"  {val_metrics['per_leadtime']['csi_m']}")

        plot_training_history(history)

        sample_x, sample_y, *sample_ex = val_ds[0]
        model.eval()
        with torch.no_grad():
            pred = model(sample_x.unsqueeze(0).to(DEVICE),
                         *[e.unsqueeze(0).to(DEVICE) for e in sample_ex])[0].cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)   # no sigmoid: clip the output

        save_comparison_gif(sample_y[:, 0].numpy(), pred[:, 0])
        print(f"\nTamamlandı.\n  Model: {MODEL_SAVE_PATH}\n  Figürler: {FIG_DIR}")
