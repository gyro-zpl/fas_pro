#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件：自定义模型转换参数
"""

# 模型路径配置
BASE_MODEL_PATH = '/home/cyyuan/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_PATH = '/home/cyyuan/FAS_PRO/sft/sft_models/test3/checkpoint-2350'
OUTPUT_PATH = '/home/cyyuan/FAS_PRO/sft/merged_model/gss3'

# 量化配置
QUANTIZATION_OPTIONS = {
    'q4_0': {'size': '~4.5GB', 'speed': '快', 'quality': '良好'},
    'q4_1': {'size': '~5GB', 'speed': '快', 'quality': '更好'},
    'q5_0': {'size': '~5.5GB', 'speed': '中等', 'quality': '很好'},
    'q5_1': {'size': '~6GB', 'speed': '中等', 'quality': '很好'},
    'q8_0': {'size': '~8GB', 'speed': '慢', 'quality': '极好'},
}

# 推荐Q4_0用于生产环境，平衡大小和性能
QUANTIZATION = "q4_0"

# Ollama模型配置
MODEL_NAME = "llama-gss-v3:8B"
SYSTEM_PROMPT = "You are a professional sociologist, proficient in simulation and prediction of social behavior.Meanwhile you are a helpful assistant."

# 生成参数
GENERATION_PARAMS = {
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'repeat_penalty': 1.1,
    'num_ctx': 4096
}

# 高级配置
ENABLE_AUTO_CLEANUP = True  # 自动清理中间文件
ENABLE_MODEL_TEST = True    # 启用模型测试
MAX_SHARD_SIZE = "5GB"      # HuggingFace模型分片大小

# llama.cpp路径配置 (如果您已安装llama.cpp，请指定路径)
LLAMA_CPP_PATH = "/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp"  # 设置为None让脚本自动检测，或指定具体路径如 "/path/to/llama.cpp"

def print_config():
    """打印当前配置"""
    print("📋 当前配置:")
    print(f"   基础模型: {BASE_MODEL_PATH.split('/')[-1]}")
    print(f"   LoRA路径: {LORA_PATH}")
    print(f"   输出路径: {OUTPUT_PATH}")
    print(f"   量化格式: {QUANTIZATION} ({QUANTIZATION_OPTIONS[QUANTIZATION]['size']}, {QUANTIZATION_OPTIONS[QUANTIZATION]['quality']})")
    print(f"   模型名称: {MODEL_NAME}")
    print(f"   系统提示: {SYSTEM_PROMPT}")

if __name__ == "__main__":
    print_config()
