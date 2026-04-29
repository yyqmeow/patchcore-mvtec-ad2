# 安装指南

## Windows 系统安装步骤

### 1. 安装 Python 依赖

在项目根目录下运行：

```powershell
pip install -r requirements.txt
```

### 2. 如果遇到网络问题，可以使用国内镜像源

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 单独安装缺失的包

如果某些包安装失败，可以单独安装：

```powershell
# 安装 timm（用于预训练模型）
pip install timm

# 安装 faiss（用于快速相似性搜索）
pip install faiss-cpu

# 如果使用 GPU，可以安装 faiss-gpu（需要先安装 CUDA）
# pip install faiss-gpu

# 安装其他依赖
pip install torch torchvision
pip install scikit-learn scikit-image scipy
pip install matplotlib pillow tqdm click
```

### 4. 验证安装

运行以下命令验证关键包是否安装成功：

```powershell
python -c "import torch; import timm; import faiss; import sklearn; print('All packages installed successfully!')"
```

### 5. 设置 PYTHONPATH

在 PowerShell 中设置环境变量：

```powershell
$env:PYTHONPATH="src"
```

或者在每次运行命令前设置：

```powershell
$env:PYTHONPATH="src"; python bin/train_vial_single.py --data_path ./vial --results_path ./results --gpu 0 --backbone wideresnet50 --imagesize 224 --save_model
```

## 常见问题

### 问题 1: ModuleNotFoundError: No module named 'timm'

**解决方法**：
```powershell
pip install timm
```

### 问题 2: ModuleNotFoundError: No module named 'faiss'

**解决方法**：
```powershell
# CPU 版本
pip install faiss-cpu

# 或使用 GPU 版本（需要 CUDA）
pip install faiss-gpu
```

### 问题 3: 安装 torch 失败

**解决方法**：
访问 PyTorch 官网获取适合你系统的安装命令：
https://pytorch.org/get-started/locally/

例如，对于 Windows + CPU：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

对于 Windows + CUDA 11.8：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题 4: 权限错误

**解决方法**：
使用管理员权限运行 PowerShell，或者使用 `--user` 参数：
```powershell
pip install --user -r requirements.txt
```

### 问题 5: 网络超时

**解决方法**：
使用国内镜像源：
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或者使用其他镜像源：
- 阿里云：https://mirrors.aliyun.com/pypi/simple/
- 豆瓣：https://pypi.douban.com/simple/
- 中科大：https://pypi.mirrors.ustc.edu.cn/simple/

## 完整安装步骤（推荐）

```powershell
# 1. 进入项目目录
cd C:\Users\Administrator\Desktop\patchcore-inspection-main

# 2. 安装依赖（使用清华镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 如果 timm 安装失败，单独安装
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 设置 PYTHONPATH
$env:PYTHONPATH="src"

# 5. 验证安装
python -c "import patchcore; print('PatchCore installed successfully!')"
```

## 检查已安装的包

```powershell
pip list | findstr -i "torch timm faiss sklearn scikit"
```

应该能看到：
- torch
- torchvision
- timm
- faiss-cpu 或 faiss-gpu
- scikit-learn
- scikit-image

## 如果还是有问题

1. **检查 Python 版本**：需要 Python 3.8 或更高版本
   ```powershell
   python --version
   ```

2. **检查 pip 版本**：确保 pip 是最新版本
   ```powershell
   python -m pip install --upgrade pip
   ```

3. **创建虚拟环境**（推荐）：
   ```powershell
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境
   .\venv\Scripts\Activate.ps1
   
   # 安装依赖
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

