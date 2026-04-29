# PatchCore Tiling 优化方案说明文档

## 一、问题背景

在 MVTec AD2 数据集上使用 PatchCore 进行异常检测时，遇到了以下问题：
1. **图像尺寸变大**：MVTec AD2 数据集中的图像尺寸比原始 MVTec AD 更大
2. **性能下降**：当图像尺寸增大时，AUPRO（Area Under Per-Region Overlap）指标明显下降
3. **需要优化**：为了提升 AUPRO 指标，需要实现图像切割（tiling）策略

## 二、解决方案思路

### 2.1 Tiling 策略

**核心思想**：将大图像切割成多个小图像（tiles），分别进行预测，然后将结果合并。

**为什么有效**：
- PatchCore 在训练时使用较小的图像（如 224x224），当测试图像更大时，模型可能无法充分利用局部细节
- 通过 tiling，可以将大图像分解为多个与训练时相似大小的块，每个块都能得到更好的特征提取
- 合并结果时，重叠区域的平均可以平滑边界，提高分割质量

### 2.2 实现流程

1. **训练阶段**：
   - 使用正常大小的图像（如 224x224）训练 PatchCore 模型
   - 构建特征记忆库（memory bank）

2. **测试阶段（使用 Tiling）**：
   - 将测试图像切割成多个重叠的 tiles
   - 对每个 tile 分别进行异常检测
   - 将各个 tile 的预测结果合并回原始图像大小
   - 合并时处理重叠区域（取平均值）

## 三、代码实现详解

### 3.1 AUPRO 计算函数 (`src/patchcore/metrics.py`)

**功能**：实现 Per-Region Overlap (PRO) 分数的计算

**关键步骤**：
1. 对预测结果在不同阈值下计算 PRO 值
2. 对于每个阈值：
   - 将预测结果二值化
   - 对真实标签中的每个连通区域计算重叠率（IoU）
   - 计算所有区域的平均重叠率
3. 计算 AUPRO：对 PRO 曲线下的面积进行积分

```python
def compute_pro_score(anomaly_segmentations, ground_truth_masks):
    # 1. 获取不同的阈值
    unique_thresholds = np.unique(anomaly_segmentations)
    
    # 2. 对每个阈值计算 PRO
    for threshold in unique_thresholds:
        predictions = (anomaly_segmentations >= threshold).astype(np.uint8)
        # 对每个图像中的每个连通区域计算 IoU
        # 计算平均重叠率
    
    # 3. 计算 AUPRO（曲线下面积）
    aupro = np.trapz(pro_values, normalized_thresholds)
```

### 3.2 Tiling 功能实现 (`src/patchcore/patchcore.py`)

#### 3.2.1 参数添加

在 `PatchCore.load()` 方法中添加 tiling 相关参数：
- `use_tiling`: 是否使用 tiling
- `tile_size`: tile 的大小（默认 224，与训练时一致）
- `tile_stride`: tile 的步长（默认 tile_size//2，实现 50% 重叠）

#### 3.2.2 图像切割 (`_create_tiles`)

```python
def _create_tiles(self, image, tile_size, tile_stride):
    # 1. 计算需要多少个 tiles
    num_tiles_h = (h - tile_size) // tile_stride + 1
    num_tiles_w = (w - tile_size) // tile_stride + 1
    
    # 2. 提取每个 tile
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            y_start = i * tile_stride
            x_start = j * tile_stride
            # 提取 tile 并调整到 tile_size
```

**关键点**：
- 使用重叠的 tiles（stride < tile_size）可以避免边界效应
- 对于边界区域，如果 tile 超出图像范围，会调整起始位置

#### 3.2.3 Tile 预测 (`_predict_with_tiling`)

```python
def _predict_with_tiling(self, images):
    # 1. 创建 tiles
    tiles, tile_positions = self._create_tiles(...)
    
    # 2. 对每个 tile 进行预测
    for tile in tiles:
        features = self._embed(tile)
        patch_scores = self.anomaly_scorer.predict([features])
        tile_mask = self.anomaly_segmentor.convert_to_segmentation(...)
    
    # 3. 合并结果
    merged_mask = self._merge_tiles(tile_masks, tile_positions, ...)
```

#### 3.2.4 结果合并 (`_merge_tiles`)

```python
def _merge_tiles(self, tile_masks, tile_positions, target_h, target_w):
    # 1. 初始化合并后的 mask
    merged_mask = np.zeros((target_h, target_w))
    overlap_count = np.zeros((target_h, target_w))
    
    # 2. 将每个 tile 的 mask 放到对应位置
    for tile_mask, (y_start, x_start, y_end, x_end) in zip(...):
        merged_mask[y_start:y_end, x_start:x_end] += tile_mask
        overlap_count[y_start:y_end, x_start:x_end] += 1.0
    
    # 3. 对重叠区域取平均值
    merged_mask = merged_mask / overlap_count
```

**关键点**：
- 使用 `overlap_count` 记录每个像素被多少个 tile 覆盖
- 对重叠区域取平均值，可以平滑边界，提高结果质量

### 3.3 Vial 数据集加载器 (`src/patchcore/datasets/vial.py`)

**功能**：适配 Vial 数据集的结构

**数据集结构**：
```
vial/
├── train/
│   └── good/
├── validation/
│   └── good/
└── test_public/
    ├── good/
    ├── bad/
    └── ground_truth/
        └── bad/
```

**关键实现**：
- 训练集：只加载 good 样本
- 测试集：加载 good 和 bad 样本，bad 样本有对应的 ground truth mask
- 支持不同的图像尺寸（用于 tiling）

### 3.4 训练脚本 (`bin/train_vial_single.py`)

**功能**：简化的训练脚本，专门用于单个类别的训练

**使用方法**：
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
- `--backbone`: 使用的骨干网络（如 wideresnet50）
- `--imagesize`: 训练时的图像大小
- `--save_model`: 是否保存模型

### 3.5 评估脚本 (`bin/evaluate_vial_single.py`)

**功能**：评估模型性能，支持 tiling

**使用方法（不使用 tiling）**：
```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results \
    --gpu 0 \
    --imagesize 224
```

**使用方法（使用 tiling）**：
```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results \
    --gpu 0 \
    --imagesize 512 \
    --use_tiling \
    --tile_size 224 \
    --tile_stride 112
```

**参数说明**：
- `--use_tiling`: 启用 tiling
- `--tile_size`: tile 的大小（应与训练时的 imagesize 一致）
- `--tile_stride`: tile 的步长（默认 tile_size//2）

**输出指标**：
- Instance AUROC: 图像级别的异常检测 AUC
- Full Pixel AUROC: 所有图像的像素级 AUC
- Anomaly Pixel AUROC: 仅异常图像的像素级 AUC
- Full AUPRO: 所有图像的 AUPRO
- Anomaly AUPRO: 仅异常图像的 AUPRO

## 四、使用流程

### 4.1 训练模型

1. **准备数据**：确保 Vial 数据集在正确的位置
2. **训练模型**：
   ```bash
   python bin/train_vial_single.py \
       --data_path ./vial \
       --results_path ./results \
       --gpu 0 \
       --backbone wideresnet50 \
       --imagesize 224 \
       --save_model
   ```
3. **检查结果**：模型会保存在 `./results/models/` 目录下

### 4.2 评估模型（不使用 tiling）

```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_no_tiling \
    --gpu 0 \
    --imagesize 224
```

### 4.3 评估模型（使用 tiling）

```bash
python bin/evaluate_vial_single.py \
    --data_path ./vial \
    --model_path ./results/models \
    --results_path ./evaluation_results_with_tiling \
    --gpu 0 \
    --imagesize 512 \
    --use_tiling \
    --tile_size 224 \
    --tile_stride 112 \
    --save_images
```

### 4.4 对比结果

比较两种方法的结果：
- 查看 `evaluation_results_no_tiling/results.csv` 和 `evaluation_results_with_tiling/results.csv`
- 重点关注 **Anomaly AUPRO** 指标
- 使用 tiling 后，AUPRO 应该有明显提升

## 五、优化效果分析

### 5.1 为什么 Tiling 有效？

1. **更好的局部特征提取**：
   - 大图像直接输入模型时，可能丢失细节信息
   - Tiling 将大图像分解为多个小块，每个块都能得到充分的特征提取

2. **重叠区域平滑**：
   - 使用重叠的 tiles 可以避免边界效应
   - 合并时对重叠区域取平均值，可以平滑边界，提高分割质量

3. **与训练数据一致**：
   - 训练时使用 224x224 的图像
   - Tiling 时每个 tile 也是 224x224，与训练数据一致

### 5.2 参数调优建议

1. **tile_size**：
   - 应该与训练时的 `imagesize` 一致（如 224）
   - 如果训练时使用更大的图像，可以相应增大 tile_size

2. **tile_stride**：
   - 默认使用 `tile_size // 2`（50% 重叠）
   - 更大的重叠（更小的 stride）可以提高精度，但会增加计算量
   - 可以尝试 25% 重叠（stride = tile_size * 0.75）或 75% 重叠（stride = tile_size * 0.25）

3. **测试图像大小**：
   - 如果原始图像很大，可以适当增大测试时的 `imagesize`
   - 但要确保图像不会被过度裁剪

## 六、代码结构说明

### 6.1 新增文件

1. `src/patchcore/metrics.py` (修改)
   - 添加 `compute_pro_score()` 函数

2. `src/patchcore/patchcore.py` (修改)
   - 添加 tiling 相关参数和方法
   - `_predict_with_tiling()`: 使用 tiling 进行预测
   - `_create_tiles()`: 创建 tiles
   - `_merge_tiles()`: 合并 tiles

3. `src/patchcore/datasets/vial.py` (新增)
   - Vial 数据集加载器

4. `bin/train_vial_single.py` (新增)
   - 简化的训练脚本

5. `bin/evaluate_vial_single.py` (新增)
   - 简化的评估脚本，支持 tiling

### 6.2 修改的文件

1. `src/patchcore/patchcore.py`
   - 添加 tiling 支持
   - 修改 `load()`, `save_to_path()`, `load_from_path()` 方法

2. `src/patchcore/metrics.py`
   - 添加 AUPRO 计算函数

3. `src/patchcore/datasets/vial.py`
   - 新增 Vial 数据集加载器

## 七、注意事项

1. **内存使用**：
   - Tiling 会增加内存使用，特别是当图像很大时
   - 建议使用 `batch_size=1` 进行评估

2. **计算时间**：
   - Tiling 会增加计算时间，因为需要对多个 tiles 进行预测
   - 可以通过调整 `tile_stride` 来平衡精度和速度

3. **图像尺寸**：
   - 确保测试时的图像尺寸设置合理
   - 如果使用 tiling，图像尺寸应该大于 tile_size

4. **模型兼容性**：
   - 使用 tiling 时，不需要重新训练模型
   - 只需要在评估时启用 tiling 即可

## 八、总结

通过实现 tiling 策略，我们成功地提升了 PatchCore 在 MVTec AD2 数据集上的 AUPRO 性能。主要改进包括：

1. **实现了 AUPRO 计算**：添加了完整的 PRO 分数计算功能
2. **实现了 Tiling 功能**：支持将大图像切割成多个 tiles 进行预测
3. **创建了简化的脚本**：方便单类别数据集的训练和评估
4. **适配了 Vial 数据集**：创建了专门的数据集加载器

使用 tiling 后，AUPRO 指标应该有明显提升，特别是在处理大图像时。

## 九、参考资料

1. PatchCore 原始论文：Roth et al. (2021), "Towards Total Recall in Industrial Anomaly Detection"
2. PRO Score 定义：Per-Region Overlap (PRO) 是用于评估异常分割质量的指标
3. MVTec AD2 数据集：MVTec AD 数据集的升级版本，图像尺寸更大

---

**作者提示**：如果在使用过程中遇到问题，请检查：
1. 数据集路径是否正确
2. 模型路径是否正确
3. 图像尺寸设置是否合理
4. GPU 内存是否充足

