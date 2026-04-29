# Windows 系统使用指南

## Windows PowerShell 命令格式

在 Windows PowerShell 中，多行命令需要使用反引号 `` ` `` 作为续行符，或者直接写成一行。

## 一、训练模型

### 方法 1：单行命令（推荐）
```powershell
python bin/train_vial_single.py --data_path ./vial --results_path ./results --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

### 方法 2：多行命令（使用反引号）
```powershell
python bin/train_vial_single.py `
    --data_path ./vial `
    --results_path ./results `
    --gpu 0 `
    --backbone wideresnet50 `
    --imagesize 224 `
    --save_model
```

## 二、评估模型（不使用 Tiling）

### 方法 1：单行命令（推荐）
```powershell
python bin/evaluate_vial_single.py --data_path ./vial --model_path ./results/models --results_path ./evaluation_results_baseline --gpu 0 --imagesize 224
```

### 方法 2：多行命令
```powershell
python bin/evaluate_vial_single.py `
    --data_path ./vial `
    --model_path ./results/models `
    --results_path ./evaluation_results_baseline `
    --gpu 0 `
    --imagesize 224
```

## 三、评估模型（使用 Tiling）

### 方法 1：单行命令（推荐）
```powershell
python bin/evaluate_vial_single.py --data_path ./vial --model_path ./results/models --results_path ./evaluation_results_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

### 方法 2：多行命令
```powershell
python bin/evaluate_vial_single.py `
    --data_path ./vial `
    --model_path ./results/models `
    --results_path ./evaluation_results_tiling `
    --gpu 0 `
    --imagesize 512 `
    --use_tiling `
    --tile_size 224 `
    --tile_stride 112 `
    --save_images
```

## 四、使用 CMD（命令提示符）

如果你使用 CMD 而不是 PowerShell，可以使用 `^` 作为续行符：

### 训练模型
```cmd
python bin/train_vial_single.py ^
    --data_path ./vial ^
    --results_path ./results ^
    --gpu 0 ^
    --backbone wideresnet50 ^
    --imagesize 224 ^
    --save_model
```

### 评估模型（使用 Tiling）
```cmd
python bin/evaluate_vial_single.py ^
    --data_path ./vial ^
    --model_path ./results/models ^
    --results_path ./evaluation_results_tiling ^
    --gpu 0 ^
    --imagesize 512 ^
    --use_tiling ^
    --tile_size 224 ^
    --tile_stride 112 ^
    --save_images
```

## 五、路径格式说明

在 Windows 中，路径可以使用：
- 正斜杠：`./vial` 或 `./results/models`
- 反斜杠（需要转义或使用原始字符串）：`.\vial` 或 `.\results\models`
- 绝对路径：`C:\Users\YourName\Desktop\patchcore-inspection-main\vial`

**推荐使用正斜杠**，Python 会自动处理。

## 六、设置 PYTHONPATH

在 PowerShell 中：
```powershell
$env:PYTHONPATH="src"
```

在 CMD 中：
```cmd
set PYTHONPATH=src
```

或者在运行命令时直接设置：
```powershell
$env:PYTHONPATH="src"; python bin/train_vial_single.py --data_path ./vial --results_path ./results --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

## 七、完整示例（推荐使用单行命令）

### 步骤 1：训练模型
```powershell
python bin/train_vial_single.py --data_path ./vial --results_path ./results --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

### 步骤 2：评估（基准）
```powershell
python bin/evaluate_vial_single.py --data_path ./vial --model_path ./results/models --results_path ./evaluation_results_baseline --gpu 0 --imagesize 224
```

### 步骤 3：评估（使用 Tiling）
```powershell
python bin/evaluate_vial_single.py --data_path ./vial --model_path ./results/models --results_path ./evaluation_results_tiling --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images
```

## 八、注意事项

1. **使用单行命令最简单**：直接复制粘贴，不需要考虑续行符
2. **路径中使用正斜杠**：`./vial` 比 `.\vial` 更通用
3. **设置 PYTHONPATH**：确保在运行前设置了 `PYTHONPATH=src`
4. **GPU 参数**：如果没有 GPU 或想使用 CPU，可以去掉 `--gpu 0` 参数

## 九、如果遇到问题

### 问题 1：找不到模块
```
ModuleNotFoundError: No module named 'patchcore'
```
**解决方法**：设置 PYTHONPATH
```powershell
$env:PYTHONPATH="src"
```

### 问题 2：路径错误
**解决方法**：使用绝对路径或检查路径是否正确
```powershell
python bin/train_vial_single.py --data_path "C:\Users\Administrator\Desktop\patchcore-inspection-main\vial" --results_path "C:\Users\Administrator\Desktop\patchcore-inspection-main\results" --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

### 问题 3：GPU 不可用
**解决方法**：去掉 `--gpu 0` 参数，使用 CPU
```powershell
python bin/train_vial_single.py --data_path ./vial --results_path ./results --backbone wideresnet50 --imagesize 224 --save_model
```

