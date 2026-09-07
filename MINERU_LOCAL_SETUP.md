# MinerU 本地安装（RTX 5070）

适用环境：Windows PowerShell、Conda `rag`、Python 3.11、RTX 5070。RTX 5070 使用 PyTorch `cu128`，Game Ready 驱动可以正常运行 CUDA。

## 1. 检查环境

```powershell
nvidia-smi
conda activate rag
python --version
```

## 2. 安装 MinerU

```powershell
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
python -m pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
mineru --version
```

安装程序不会自动下载 MinerU 模型。

## 3. 安装 CUDA 版 PyTorch 和 lmdeploy

先删除可能被普通 PyPI 镜像安装的 CPU 版 PyTorch：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

安装适用于 Windows、Python 3.11 和 RTX 50 系列的 lmdeploy：

```powershell
$lmdeployWheel = "https://github.com/InternLM/lmdeploy/releases/download/v0.11.1/lmdeploy-0.11.1+cu128-cp311-cp311-win_amd64.whl"
python -m pip install --force-reinstall $lmdeployWheel --no-dependencies
```

LMDeploy PyTorch 后端在 Windows 上还需要 Triton。PyTorch 2.8 对应
Triton 3.4：

```powershell
python -m pip install "triton-windows==3.4.0.post21" -i https://pypi.org/simple
python -m lmdeploy.pytorch.check_env.triton_custom_add
```

自检成功时输出：`Done.`。

验证 GPU：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU可用:', torch.cuda.is_available()); print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

预期包含：`2.8.0+cu128`、`12.8`、`True` 和 `NVIDIA GeForce RTX 5070`。

RTX 5070 的计算能力是 `sm_120`，当前 MinerU 3.4.5 所使用的
`lmdeploy 0.11.1` TurboMind 预编译内核不兼容。将 `rag` 环境固定为
LMDeploy PyTorch 后端（仍然使用 GPU）：

```powershell
conda env config vars set -n rag MINERU_LMDEPLOY_BACKEND=pytorch TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
conda deactivate
conda activate rag
```

`TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` 用于避开 Windows 下 PyTorch
2.8 静态 CUDA launcher 的 `Python int too large to convert to C long`
溢出错误。

验证配置：

```powershell
echo $env:MINERU_LMDEPLOY_BACKEND
echo $env:TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER
```

预期分别输出：`pytorch` 和 `0`。

## 4. 下载 MinerU VLM 模型

模型保存到 `E:\ai_models\modelscope_cache`：

```powershell
New-Item -ItemType Directory -Force -Path E:\ai_models\modelscope_cache
$env:MODELSCOPE_CACHE="E:\ai_models\modelscope_cache"
$env:MINERU_MODEL_SOURCE="modelscope"
$env:MODELSCOPE_OFFLINE="0"

mineru-models-download -s modelscope -m vlm
```

下载完成后启用离线模式：

```powershell
$env:MODELSCOPE_OFFLINE="1"
```

`$env:...` 只对当前 PowerShell 会话有效，重新打开终端后需要再次设置。

## 5. 运行 MinerU

直接解析 PDF：

```powershell
New-Item -ItemType Directory -Force -Path E:\code\py\rag\output
mineru -p "E:\路径\输入文件.pdf" -o "E:\code\py\rag\output" -b vlm-engine
```

日志中应显示：

```text
lmdeploy device is: cuda, lmdeploy backend is: pytorch
```

启动本地 API：

```powershell
mineru-api --host 127.0.0.1 --port 8000 --enable-vlm-preload true
```

API 文档：`http://127.0.0.1:8000/docs`
