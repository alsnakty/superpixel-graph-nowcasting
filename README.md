# Superpixel graph attention on a Lagrangian base for radar precipitation nowcasting

This repository contains the code of LSP-GAT (Lagrangian Superpixel Graph Attention Network) and of the forecasts it is compared with: a no-graph control, a ConvLSTM, Lagrangian persistence and Eulerian persistence. The data are the MeteoNet rain radar maps for north-western France.

LSP-GAT forecasts the six radar maps of the next hour, at 10 min steps, as Lagrangian persistence plus a learned correction. A recurrent graph attention network produces the correction, and its nodes are superpixels segmented from the last observed map.

## Files

| File | Role |
|---|---|
| `meteonet_preprocess.py` | Turns the MeteoNet radar files into 128 x 128 sequences of 25 maps |
| `analysis/lagrange_kalicilik.py` | Optical flow and semi-Lagrangian advection (Lagrangian persistence) |
| `analysis/akis_onhesap.py` | Precomputes the motion field of every sequence |
| `analysis/akis_manifest.py` | Records that tie a motion-field file to its data file |
| `analysis/veri_secimi.py` | Selects the sequences of the training, validation and test sets |
| `analysis/split/strict_split_selection.json` | The selected sequences (8,392 / 2,080 / 2,114) |
| `meteonet_sp_gat_lagrange.py` | LSP-GAT |
| `meteonet_sp_bagimsiz_lagrange.py` | No-graph control |
| `meteonet_baseline_convlstm.py` | ConvLSTM |
| `analysis/ortak_degerlendirme.py` | Scores the ConvLSTM and the two persistence forecasts |
| `analysis/ortak_degerlendirme_sp.py` | Scores LSP-GAT and the no-graph control on the validation set |
| `analysis/ortak_degerlendirme_sp_test.py` | Scores LSP-GAT and the no-graph control on the test set |

Identifiers and console messages are in Turkish. The model and scoring files also contain switches and entries for variants that are not part of the paper. They are off by default: leave `MOTION_EDGES`, `MOTION_SIGN` and `HORIZON_GAT` unset.

## Requirements

Python 3.10 with the packages in `requirements.txt`: PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, scikit-image, Matplotlib and imageio.

## Data

MeteoNet is released by Météo-France: <https://meteofrance.github.io/meteonet/>. Download the rain radar archives of the north-western zone for 2016 to 2018 and unpack them. Each yearly archive contains monthly archives, which must be unpacked as well.

## Usage

All settings are environment variables. The commands below use the settings of the paper.

### 1. Sequences

```bash
DOWNSAMPLE=blockmax INPUT_ROOT=<folder with the unpacked .npz files> OUTPUT_DIR=./data python meteonet_preprocess.py
```

This writes `meteonet_nw_train.npy`, `meteonet_nw_val.npy` and `meteonet_nw_test.npy` to `./data`.

### 2. Data check and motion fields

```bash
DATA_DIR=./data python analysis/veri_secimi.py
FLOW=tvl1 DATA_DIR=./data python analysis/akis_onhesap.py train val test
AKIS_HASH_DOGRULA=1 DATA_DIR=./data python analysis/akis_manifest.py --dogrula tvl1
```

The first command checks the three data files. The second command estimates one TV-L1 motion field per sequence. The third command checks the motion-field files. Run all three before training and scoring.

### 3. Training

```bash
STRICT_SPLIT=1 EVAL_TEST=0 BMSE_WEIGHTS=1,1,1,1,1,1,1 TAG_SUFFIX=_mse SEED=42 DATA_DIR=./data OUT_DIR=./outputs python meteonet_sp_gat_lagrange.py
```

Use the same command with `meteonet_sp_bagimsiz_lagrange.py` for the no-graph control and with `meteonet_baseline_convlstm.py` for the ConvLSTM. The paper uses the seeds 42, 123 and 456. `BMSE_WEIGHTS=1,1,1,1,1,1,1` selects the plain mean squared error. Each run writes a checkpoint (`.pt`) and a record (`_metrics.json`) to `OUT_DIR`; the scoring scripts read both.

The two superpixel models have built-in checks that run on small synthetic tensors and need no data:

```bash
SELF_TEST=1 python meteonet_sp_gat_lagrange.py
```

### 4. Scores

```bash
MODEL=SP_GAT SPLIT=val DATA_DIR=./data CKPT=./outputs/sp_gat_lagrange_res_mse_seed42.pt python analysis/ortak_degerlendirme_sp.py
MODEL=SP_BAG SPLIT=val DATA_DIR=./data CKPT=./outputs/sp_bag_lagrange_res_mse_seed42.pt python analysis/ortak_degerlendirme_sp.py
MODEL=C0 SPLIT=val DATA_DIR=./data CKPT=./outputs/baseline_convlstm_mse_seed42.pt python analysis/ortak_degerlendirme.py
MODEL=L SPLIT=val DATA_DIR=./data python analysis/ortak_degerlendirme.py
MODEL=E SPLIT=val DATA_DIR=./data python analysis/ortak_degerlendirme.py
```

`SP_GAT` is LSP-GAT, `SP_BAG` is the no-graph control, `C0` is the ConvLSTM, `L` is Lagrangian persistence and `E` is Eulerian persistence. For the test set, use `SPLIT=test`, and `analysis/ortak_degerlendirme_sp_test.py` for the two superpixel models.

Each call writes a JSON file with the scores and an `.npz` file with the counts of every sequence to `analysis/`, or to the path given in `OUT`.

| Key in the JSON file | Score |
|---|---|
| `S_long_t30_60` | CSI averaged over the six thresholds and the lead times from 30 to 60 min |
| `csi_t10` | CSI at the first lead time, averaged over the six thresholds |
| `csi_at10mmh` | CSI at 10 mm/h with the counts summed over all lead times |
| `pool16_genel` | CSI after 16 x 16 max pooling |
| `kutle_orani` | Mass ratio: forecast rain total divided by observed rain total |

## License

MIT. See `LICENSE`.
