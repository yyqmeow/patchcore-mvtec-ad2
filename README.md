# PatchCore on MVTec AD 2

Investigating why PatchCore fails on the MVTec AD 2 benchmark and proposing simple modifications that close most of the pixel-level segmentation gap.

> Course project for **IMT4392 — Deep Learning** at NTNU.
> Author: **Yayang Qian** ([report PDF](report/IMT4392_Deep_Learning_Report_Yayang_Qian.pdf))

---

## Summary

State-of-the-art unsupervised anomaly detection methods like PatchCore are near-saturated on the original MVTec AD, but collapse on the harder MVTec AD 2 — particularly on pixel-level localization (AU-PRO). This project diagnoses *why* and proposes fixes.

**Key result:** Multi-scale feature fusion (ResNet50 layer 2 + layer 3) raises **mean AU-PRO from 8.87% to 76.35%** on MVTec AD 2, with category-level gains such as *fruit_jelly* 41.3% → 90.7% and *vial* 46.0% → 91.2%.

### Contributions

1. **Tiling is not the cure.** A 4×4 tiling strategy (256×256 patches over 1024×1024 inputs) yields no significant AU-PRO improvement once a train/test distribution bug is fixed (AU-PRO stays at ~9%) — refuting the resolution-mismatch hypothesis.
2. **Feature granularity is the bottleneck.** Layer3-only features give strong image-level AUROC but poor segmentation; layer2-only improves localization but produces false alarms. **Fusing layer2 + layer3 combines the strengths of both** and is the main driver of the 76.35% AU-PRO result.
3. **Image-level vs pixel-level trade-off.** Multi-scale fusion drops image AUROC from ~90% (layer3-only) to ~66% — layer2 features are more sensitive to benign texture variation. A hybrid scoring scheme (layer3-only for image-level, fused for pixel-level) is proposed as future work.
4. **Swin-T pitfall identified.** Naively applying Global Average Pooling to Swin-T output collapses spatial information and reduces PatchCore to an image-level detector (mean AU-PRO ≈ 51%). Replacing GAP with the full token feature map is required; this fix was identified but not fully re-run within the project.
5. **BYOL self-supervised pretraining does not rescue the GAP bug.** Under the broken GAP integration, supervised vs BYOL Swin-T differ by < 1% — pretraining alone cannot compensate for losing spatial structure.
6. **Memory bank size is not the bottleneck.** A coreset of 3000 patches performed slightly worse than the default 2000 (69.23% vs 76.35% AU-PRO). The contribution lies in the feature domain, not memory bank design.

See [`report/IMT4392_Deep_Learning_Report_Yayang_Qian.pdf`](report/IMT4392_Deep_Learning_Report_Yayang_Qian.pdf) for the full methodology, ablations, and qualitative results.

---

## Results

### Overall comparison on MVTec AD 2

| Backbone / config                            | Image AUROC | Pixel AUROC | AU-PRO (5%) |
|----------------------------------------------|------------:|------------:|------------:|
| WR50 (layer2, default coreset)               |       0.704 |       0.857 |       38.49 |
| ResNet18 (layer3, coreset 1000)              |       0.675 |       0.500 |       64.04 |
| ResNet50 layer3-only (single-scale baseline) |       0.900 |       0.916 |        8.87 |
| **ResNet50, L2 + L3 fused** *(this work)*    |   **0.660** |   **0.763** |   **76.35** |
| ResNet50, L2+L3, coreset 3000                |       0.582 |       0.692 |       69.23 |
| ResNet50, L2+L3, coreset 2000                |       0.593 |       0.675 |       67.58 |
| Swin-T (broken GAP integration)              |       0.476 |       0.509 |       50.95 |
| Swin-T + BYOL (broken GAP integration)       |       0.506 |       0.502 |       50.25 |

### Per-category breakdown — best model (ResNet50, L2 + L3)

| Category    | Image AUROC | Pixel AUROC | AU-PRO (5%) |
|-------------|------------:|------------:|------------:|
| Fruit Jelly |       0.783 |       0.907 |   **90.74** |
| Vial        |       0.761 |       0.912 |   **91.17** |
| Walnuts     |       0.733 |       0.815 |       81.45 |
| Wallplugs   |       0.646 |       0.780 |       78.00 |
| Rice        |       0.603 |       0.765 |       76.48 |
| Sheet Metal |       0.601 |       0.764 |       76.38 |
| Can         |       0.501 |       0.696 |       69.61 |
| Fabric      |       0.650 |       0.470 |       46.96 |
| **Mean**    |   **0.660** |   **0.763** |   **76.35** |

Vial and Fruit Jelly cross 90% AU-PRO. **Fabric remains the open challenge** — sub-pattern weave irregularities are not separable in either layer2 or layer3 ImageNet features; lower-layer texture descriptors or domain-specific pretraining are likely needed.

Aggregated XLSX/PNG summaries: [`results/summary/`](results/summary/).
Per-run logs (`evaluation.log`, `metrics_logger.json`, `readme.txt`) and per-category outputs: [`results/per-backbone/`](results/per-backbone/).

---

## Repository layout

```
patchcore-mvtec-ad2/
├── src/patchcore/          Core code (PatchCore + MVTec AD 2 / VisA dataset adapters)
├── bin/                    Training & evaluation entry points
│   ├── train_mvtec_ad2_single.py
│   ├── evaluate_mvtec_ad2_single.py
│   ├── train_vial_single.py
│   ├── evaluate_vial_single.py
│   ├── run_patchcore.py
│   └── load_and_evaluate_patchcore.py
├── tests/                  Tiling unit tests
├── scripts/                Bash / PowerShell / batch wrappers
├── notebooks/screws.ipynb  Exploratory notebook (Swin / BYOL experiments)
├── models/                 PatchCore memory banks + per-config result CSVs
├── results/
│   ├── summary/            Aggregated XLSX + result-grid PNGs
│   └── per-backbone/       Per-run logs and outputs (resnet18 / resnet50 / WR50 / Swin)
├── docs/                   Install / quickstart / tiling docs (upstream + project-specific)
└── report/                 Final project report (PDF)
```

---

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for details and CUDA setup.

## Quickstart

- General PatchCore usage: [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- MVTec AD 2 specifically: [`docs/QUICKSTART_MVTEC_AD2.md`](docs/QUICKSTART_MVTEC_AD2.md)
- Tiling experiments: [`docs/TILING_OPTIMIZATION_README.md`](docs/TILING_OPTIMIZATION_README.md)
- Windows: [`docs/WINDOWS_USAGE.md`](docs/WINDOWS_USAGE.md)

### Train on a single MVTec AD 2 category

```bash
python bin/train_mvtec_ad2_single.py \
    --data_path /path/to/mvtec_ad_2 \
    --category can \
    --output_dir results/can
```

### Evaluate a trained model

```bash
python bin/evaluate_mvtec_ad2_single.py \
    --model_path models/IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1 \
    --data_path /path/to/mvtec_ad_2 \
    --category can
```

---

## Acknowledgments & upstream

This repository builds directly on the official PatchCore implementation:

- **PatchCore** — Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022). Original code: <https://github.com/amazon-science/patchcore-inspection> ([upstream README](docs/UPSTREAM_README.md), Apache 2.0).

Reference materials consulted but **not included** here (they have their own repositories):

- **anomalib** — <https://github.com/openvinotoolkit/anomalib>
- **spot-diff** — <https://github.com/amazon-science/spot-diff>

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE), inherited from the upstream PatchCore repository.

## Citation

If this work is useful to you, please cite the upstream PatchCore paper:

```bibtex
@inproceedings{roth2022towards,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle={CVPR},
  year={2022}
}
```
