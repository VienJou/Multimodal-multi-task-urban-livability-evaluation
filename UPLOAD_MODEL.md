# Upload Model to Hugging Face Model Hub

由于模型文件太大（669MB），超过 GitHub 的 100MB 文件大小限制，建议将模型上传到 Hugging Face Model Hub。

## 方法 1: 使用 Hugging Face Hub（推荐）

### 1. 安装 huggingface_hub

```bash
pip install huggingface_hub
```

### 2. 登录 Hugging Face

```bash
huggingface-cli login
```

输入你的 Hugging Face token（可以在 https://huggingface.co/settings/tokens 获取）

### 3. 上传模型

创建一个 Python 脚本来上传模型：

```python
from huggingface_hub import HfApi, upload_folder

# 初始化 API
api = HfApi()

# 创建一个新的模型仓库（或使用现有的）
repo_id = "Vinjou/Multimodal-urban-livability-evaluation-model"  # 修改为你的用户名和仓库名

# 上传整个 livability_4M_6aspects 目录
upload_folder(
    folder_path="livability_4M_6aspects",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["*.log", "*.txt"]  # 排除日志文件
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
```

### 4. 在代码中加载模型

之后可以从 Hugging Face 加载模型：

```python
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

# 下载模型文件
model_path = hf_hub_download(
    repo_id="Vinjou/Multimodal-urban-livability-evaluation-model",
    filename="pytorch_model.bin",
    local_dir="./livability_4M_6aspects"
)
```

## 方法 2: 使用 Git LFS（需要安装）

如果你有权限安装 Git LFS：

### 1. 安装 Git LFS

```bash
# Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 或使用 conda
conda install -c conda-forge git-lfs
```

### 2. 初始化 Git LFS

```bash
git lfs install
```

### 3. 跟踪大文件

```bash
git lfs track "*.bin"
git lfs track "livability_4M_6aspects/pytorch_model.bin"
```

### 4. 添加并推送

```bash
git add .gitattributes
git add livability_4M_6aspects/pytorch_model.bin
git commit -m "Add model files with Git LFS"
git push origin main
```

**注意**：Git LFS 有存储限制（免费账户 1GB），大文件可能需要付费。

## 推荐方案

**强烈建议使用方法 1（Hugging Face Model Hub）**，因为：
- ✅ 免费，存储空间充足
- ✅ 专门为大文件设计
- ✅ 易于分享和下载
- ✅ 版本控制
- ✅ 可以直接通过 Hugging Face API 加载

