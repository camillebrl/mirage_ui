#!/usr/bin/env python3
"""Test script to diagnose model download issues."""

import os

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_huggingface_access():
    """Test if we can access Hugging Face."""
    print("🔍 Testing Hugging Face access...")
    try:
        response = requests.get("https://huggingface.co/api/models/Qwen/Qwen3-0.6B", timeout=10)
        if response.status_code == 200:
            print("✅ Can access Hugging Face API")
        else:
            print(f"❌ Hugging Face API returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach Hugging Face: {e}")

def test_model_download():
    """Test downloading a small model."""
    model_name = "Qwen/Qwen3-0.6B"
    
    print(f"\n🔍 Testing model download: {model_name}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print("\n📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True  # Required for some models
        )
        print("✅ Tokenizer downloaded successfully")
        
        print("\n📥 Downloading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,  # Required for some models
            low_cpu_mem_usage=True
        )
        print("✅ Model downloaded successfully")
        
        # Test inference
        print("\n🧪 Testing inference...")
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful: {result[:50]}...")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print("\n💡 Possible solutions:")
        print("   1. Check your internet connection")
        print("   2. Try setting HF_HUB_OFFLINE=0")
        print("   3. Login to Hugging Face: huggingface-cli login")
        print("   4. Clear cache: rm -rf ~/.cache/huggingface")

if __name__ == "__main__":
    test_huggingface_access()
    test_model_download()