#!/usr/bin/env python3
"""
Upload model to Hugging Face Model Hub
Usage: python upload_model_to_hf.py
"""

from huggingface_hub import HfApi, upload_folder
import os

def upload_model():
    # 检查是否已登录
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if not token:
        print("❌ Not logged in to Hugging Face. Please run:")
        print("   huggingface-cli login")
        return
    
    # 模型仓库 ID（修改为你的用户名和仓库名）
    repo_id = "Vinjou/Multimodal-urban-livability-evaluation-model"
    
    # 检查模型目录是否存在
    model_dir = "livability_4M_6aspects"
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print(f"❌ Model file not found in: {model_dir}")
        return
    
    print(f"📤 Uploading model to: https://huggingface.co/{repo_id}")
    print("   This may take a while (model is ~669MB)...")
    
    try:
        # 上传模型目录
        upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.log", "training_log.txt", "eval_results.txt", "final_eval_results.txt"],
            commit_message="Upload trained model"
        )
        print(f"✅ Model uploaded successfully!")
        print(f"   View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        print("
Make sure:")
        print("  1. You're logged in: huggingface-cli login")
        print("  2. You have write access to the repository")
        print("  3. The repository exists (or will be created)")

if __name__ == "__main__":
    upload_model()
