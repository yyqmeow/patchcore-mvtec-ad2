# MVTec AD2 快速开始指南

本指南提供快速使用其他类别（非 vial）的示例。

## 快速示例：训练和评估 can 类别

### 步骤 1：训练模型

```powershell
# Windows PowerShell
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --results_path ./results_can --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

```bash
# Linux/Mac
python bin/train_mvtec_ad2_single.py \
    --data_path ./mvtec_ad_2 \
    --classname can \
    --results_path ./results_can \
    --gpu 0 \
    --backbone wideresnet50 \
    --imagesize 224 \
    --save_model
```

### 步骤 2：评估模型（基准）

```powershell
# Windows PowerShell
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_baseline --gpu 0 --imagesize 224
```

### 步骤 3：评估模型（使用 Tiling）

```powershell
# Windows PowerShell
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

## 所有支持的类别

- `can` - 罐子
- `fabric` - 织物
- `fruit_jelly` - 果冻
- `rice` - 大米
- `sheet_metal` - 金属板
- `wallplugs` - 墙插
- `walnuts` - 核桃

（注意：`vial` 类别可以使用专门的 `train_vial_single.py` 和 `evaluate_vial_single.py` 脚本）

## 查看结果

评估结果保存在 `results_*.csv` 文件中，包含以下指标：
- Instance AUROC
- Full Pixel AUROC
- Anomaly Pixel AUROC
- Full AUPRO
- Anomaly AUPRO（重点关注）

## 更多信息

详细使用说明请参考 [MVTEC_AD2_USAGE.md](MVTEC_AD2_USAGE.md)

