# 快速使用指南

## 一、环境准备

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **设置 PYTHONPATH**：
   ```bash
   # Windows
   set PYTHONPATH=src
   
   # Linux/Mac
   export PYTHONPATH=src
   ```

## 二、数据准备

确保 Vial 数据集结构如下：
```
vial/
├── train/
│   └── good/
│       └── *.png
├── validation/
│   └── good/
│       └── *.png
└── test_public/
    ├── good/
    │   └── *.png
    ├── bad/
    │   └── *.png
    └── ground_truth/
        └── bad/
            └── *.png
```

## 三、训练模型

```bash
python bin/train_vial_single.py \
    --data_path ./vial \
    --results_path ./results \
    --gpu 0 \
    --backbone wideresnet50 \
    --imagesize 224 \
    --save_model
```

**参数说明**：
- `--data_path`: Vial 数据集路径
- `--results_path`: 结果保存路径
- `--gpu`: GPU ID（如果使用 CPU，不需要此参数）
- `--backbone`: 骨干网络（如 wideresnet50, resnet50 等）
- `--imagesize`: 训练图像大小（建议 224）
- `--save_model`: 保存训练好的模型

## 四、评估模型

### 4.1 不使用 Tiling（基准）

```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_baseline \
    --gpu 0 \
    --imagesize 224
```

### 4.2 使用 Tiling（优化后）

```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_tiling \
    --gpu 0 \
    --imagesize 512 \
    --use_tiling \
    --tile_size 224 \
    --tile_stride 112 \
    --save_images
```

**参数说明**：
- `--use_tiling`: 启用 tiling 功能
- `--tile_size`: Tile 大小（应与训练时的 imagesize 一致，如 224）
- `--tile_stride`: Tile 步长（默认 tile_size//2，即 50% 重叠）
- `--save_images`: 保存分割结果图像

## 五、查看结果

结果会保存在 `results.csv` 文件中，包含以下指标：
- **Instance AUROC**: 图像级别的异常检测 AUC
- **Full Pixel AUROC**: 所有图像的像素级 AUC
- **Anomaly Pixel AUROC**: 仅异常图像的像素级 AUC
- **Full AUPRO**: 所有图像的 AUPRO
- **Anomaly AUPRO**: 仅异常图像的 AUPRO（**重点关注的指标**）

## 六、预期效果

使用 Tiling 后，**Anomaly AUPRO** 应该有明显提升（通常提升 2-5%）。

## 七、常见问题

### 7.1 内存不足

如果遇到内存不足的问题：
1. 减小 `--tile_size`（如 128）
2. 增大 `--tile_stride`（减少重叠）
3. 使用 CPU 模式（不指定 `--gpu`）

### 7.2 图像尺寸不匹配

如果遇到图像尺寸不匹配的问题：
1. 检查训练和评估时的 `--imagesize` 参数
2. 使用 tiling 时，确保测试图像大小大于 tile_size
3. 检查数据集中的图像是否被正确加载

### 7.3 模型加载失败

如果模型加载失败：
1. 检查 `--model_path` 是否正确
2. 确保模型文件完整（应包含 `.faiss` 和 `.pkl` 文件）
3. 检查模型是否是用相同版本的代码训练的

## 八、参数调优建议

1. **tile_size**: 
   - 建议与训练时的 `imagesize` 一致（如 224）
   - 如果 GPU 内存充足，可以尝试更大的值（如 256）

2. **tile_stride**:
   - 默认 `tile_size // 2`（50% 重叠）通常效果最好
   - 可以尝试 25% 重叠（`tile_size * 0.75`）以提高速度
   - 可以尝试 75% 重叠（`tile_size * 0.25`）以提高精度

3. **测试图像大小**:
   - 如果原始图像很大，可以适当增大 `--imagesize`
   - 但要确保图像不会被过度裁剪
   - 建议使用 tiling 时，`imagesize` 应该大于 `tile_size`

## 九、完整示例

### 训练 + 评估（不使用 Tiling）

```bash
# 1. 训练
python bin/train_vial_single.py \
    --data_path ./vial \
    --results_path ./results \
    --gpu 0 \
    --backbone wideresnet50 \
    --imagesize 224 \
    --save_model

# 2. 评估（基准）
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_baseline \
    --gpu 0 \
    --imagesize 224
```

### 训练 + 评估（使用 Tiling）

```bash
# 1. 训练（与上面相同）
python bin/train_vial_single.py \
    --data_path ./vial \
    --results_path ./results \
    --gpu 0 \
    --backbone wideresnet50 \
    --imagesize 224 \
    --save_model

# 2. 评估（使用 Tiling）
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_tiling \
    --gpu 0 \
    --imagesize 512 \
    --use_tiling \
    --tile_size 224 \
    --tile_stride 112 \
    --save_images
```

## 十、结果对比

比较两种方法的结果：
```bash
# 查看基准结果
cat evaluation_results_baseline/results.csv

# 查看 Tiling 结果
cat evaluation_results_tiling/results.csv
```

重点关注 **Anomaly AUPRO** 的提升！

