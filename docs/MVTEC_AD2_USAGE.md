# MVTec AD2 数据集使用指南

本指南说明如何使用 PatchCore 在 MVTec AD2 数据集的其他类别上进行训练和评估。

## 支持的类别

MVTec AD2 数据集包含以下 8 个类别：
- `can` - 罐子
- `fabric` - 织物
- `fruit_jelly` - 果冻
- `rice` - 大米
- `sheet_metal` - 金属板
- `vial` - 小瓶（已有专门脚本）
- `wallplugs` - 墙插
- `walnuts` - 核桃

## 一、训练模型

### Windows PowerShell 命令

```powershell
# 训练 can 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --results_path ./results_can --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 fabric 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fabric --results_path ./results_fabric --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 fruit_jelly 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fruit_jelly --results_path ./results_fruit_jelly --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 rice 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname rice --results_path ./results_rice --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 sheet_metal 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname sheet_metal --results_path ./results_sheet_metal --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 wallplugs 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname wallplugs --results_path ./results_wallplugs --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 训练 walnuts 类别
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname walnuts --results_path ./results_walnuts --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

### Linux/Mac 命令

```bash
# 训练 can 类别
python bin/train_mvtec_ad2_single.py \
    --data_path ./mvtec_ad_2 \
    --classname can \
    --results_path ./results_can \
    --gpu 0 \
    --backbone wideresnet50 \
    --imagesize 224 \
    --save_model
```

## 二、评估模型（不使用 Tiling）

### Windows PowerShell 命令

```powershell
# 评估 can 类别（基准）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_baseline --gpu 0 --imagesize 224

# 评估 fabric 类别（基准）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fabric --model_path ./results_fabric/models/fabric --results_path ./evaluation_results_fabric_baseline --gpu 0 --imagesize 224
```

## 三、评估模型（使用 Tiling）

### Windows PowerShell 命令

```powershell
# 评估 can 类别（使用 Tiling）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images

# 评估 fabric 类别（使用 Tiling）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fabric --model_path ./results_fabric/models/fabric --results_path ./evaluation_results_fabric_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

## 四、批量训练所有类别

### Windows PowerShell 脚本

创建一个 `train_all_classes.ps1` 文件：

```powershell
$classes = @("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "wallplugs", "walnuts")
$dataPath = "./mvtec_ad_2"
$gpu = 0

foreach ($class in $classes) {
    Write-Host "Training class: $class"
    python bin/train_mvtec_ad2_single.py `
        --data_path $dataPath `
        --classname $class `
        --results_path "./results_$class" `
        --gpu $gpu `
        --backbone wideresnet50 `
        --imagesize 224 `
        --save_model
    Write-Host "Completed training for $class"
}
```

运行：
```powershell
.\train_all_classes.ps1
```

### Linux/Mac Bash 脚本

创建一个 `train_all_classes.sh` 文件：

```bash
#!/bin/bash

classes=("can" "fabric" "fruit_jelly" "rice" "sheet_metal" "wallplugs" "walnuts")
data_path="./mvtec_ad_2"
gpu=0

for class in "${classes[@]}"; do
    echo "Training class: $class"
    python bin/train_mvtec_ad2_single.py \
        --data_path $data_path \
        --classname $class \
        --results_path "./results_$class" \
        --gpu $gpu \
        --backbone wideresnet50 \
        --imagesize 224 \
        --save_model
    echo "Completed training for $class"
done
```

运行：
```bash
chmod +x train_all_classes.sh
./train_all_classes.sh
```

## 五、批量评估所有类别

### Windows PowerShell 脚本

创建一个 `evaluate_all_classes.ps1` 文件：

```powershell
$classes = @("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "wallplugs", "walnuts")
$dataPath = "./mvtec_ad_2"
$gpu = 0

# 评估基准（不使用 Tiling）
foreach ($class in $classes) {
    Write-Host "Evaluating class: $class (baseline)"
    python bin/evaluate_mvtec_ad2_single.py `
        --data_path $dataPath `
        --classname $class `
        --model_path "./results_$class/models/$class" `
        --results_path "./evaluation_results_${class}_baseline" `
        --gpu $gpu `
        --imagesize 224
}

# 评估 Tiling（使用 Tiling）
foreach ($class in $classes) {
    Write-Host "Evaluating class: $class (with tiling)"
    python bin/evaluate_mvtec_ad2_single.py `
        --data_path $dataPath `
        --classname $class `
        --model_path "./results_$class/models/$class" `
        --results_path "./evaluation_results_${class}_tiling" `
        --gpu $gpu `
        --imagesize 512 `
        --use_tiling `
        --tile_size 224 `
        --tile_stride 112 `
        --save_images
}
```

## 六、参数说明

### 训练参数

- `--data_path`: MVTec AD2 数据集根目录路径（如 `./mvtec_ad_2`）
- `--classname`: 类别名称（必须从支持的类别中选择）
- `--results_path`: 结果保存路径
- `--gpu`: GPU ID（如果使用 CPU，可以去掉此参数）
- `--backbone`: 骨干网络（如 `wideresnet50`, `resnet50` 等）
- `--imagesize`: 训练图像大小（建议 224）
- `--save_model`: 保存训练好的模型

### 评估参数

- `--data_path`: MVTec AD2 数据集根目录路径
- `--classname`: 类别名称
- `--model_path`: 训练好的模型路径（应指向包含 `.faiss` 和 `.pkl` 文件的目录）
- `--results_path`: 结果保存路径
- `--use_tiling`: 启用 tiling 功能
- `--tile_size`: Tile 大小（应与训练时的 imagesize 一致，如 224）
- `--tile_stride`: Tile 步长（默认 tile_size//2，即 50% 重叠）
- `--save_images`: 保存分割结果图像

## 七、结果查看

每个类别的评估结果会保存在对应的 `results_*.csv` 文件中，包含以下指标：

- **Instance AUROC**: 图像级别的异常检测 AUC
- **Full Pixel AUROC**: 所有图像的像素级 AUC
- **Anomaly Pixel AUROC**: 仅异常图像的像素级 AUC
- **Full AUPRO**: 所有图像的 AUPRO
- **Anomaly AUPRO**: 仅异常图像的 AUPRO（**重点关注的指标**）

## 八、完整示例

### 示例 1：训练和评估 can 类别

```powershell
# 1. 训练
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --results_path ./results_can --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 2. 评估（基准）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_baseline --gpu 0 --imagesize 224

# 3. 评估（使用 Tiling）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

### 示例 2：训练和评估 fabric 类别

```powershell
# 1. 训练
python bin/train_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fabric --results_path ./results_fabric --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model

# 2. 评估（使用 Tiling）
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname fabric --model_path ./results_fabric/models/fabric --results_path ./evaluation_results_fabric_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

## 九、注意事项

1. **数据集路径**：确保 `--data_path` 指向 MVTec AD2 数据集的根目录（包含所有类别文件夹的目录）
2. **模型路径**：`--model_path` 应指向包含模型文件的目录（如 `./results_can/models/can`），该目录应包含 `nnscorer_search_index.faiss` 和 `patchcore_params.pkl` 文件
3. **类别名称**：必须使用正确的类别名称（区分大小写）
4. **GPU 内存**：如果遇到内存不足，可以减小 `--tile_size` 或增大 `--tile_stride`
5. **PYTHONPATH**：确保设置了 `PYTHONPATH=src`（Windows: `$env:PYTHONPATH="src"`，Linux/Mac: `export PYTHONPATH=src`）

## 十、常见问题

### 问题 1：找不到模块

```
ModuleNotFoundError: No module named 'patchcore'
```

**解决方法**：设置 PYTHONPATH
```powershell
$env:PYTHONPATH="src"
```

### 问题 2：类别名称错误

```
Invalid classname: xxx
```

**解决方法**：检查类别名称是否正确，支持的类别：`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`

### 问题 3：模型路径错误

```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方法**：确保模型路径正确，应指向包含 `.faiss` 和 `.pkl` 文件的目录

---

**提示**：使用 Tiling 后，**Anomaly AUPRO** 指标通常会有明显提升（2-5%），特别是在处理大图像时。

