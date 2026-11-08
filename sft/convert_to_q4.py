#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将已合并的HuggingFace模型转换为Q4量化的GGUF格式
"""

import os
import subprocess
from pathlib import Path

# 配置路径
MERGED_MODEL_PATH = "/home/cyyuan/FAS_PRO/sft/merged_model/gss3"
OUTPUT_DIR = "/home/cyyuan/FAS_PRO/sft/merged_model/gss3_q4"
LLAMA_CPP_DIR = "/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp"
MODEL_NAME = "llama-gss-v3:8B"
def check_tools():
    """检查所需工具"""
    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    quantize_tool = Path(LLAMA_CPP_DIR) / "build/bin/llama-quantize"
    
    print("🔍 检查工具...")
    print(f"转换脚本: {convert_script}")
    print(f"量化工具: {quantize_tool}")
    
    if not convert_script.exists():
        print(f"❌ 转换脚本不存在: {convert_script}")
        return None, None
    
    if not quantize_tool.exists():
        print(f"❌ 量化工具不存在: {quantize_tool}")
        return convert_script, None
    
    print("✅ 工具检查完成")
    return convert_script, quantize_tool

def convert_to_fp16_gguf(model_path, output_dir, convert_script):
    """第一步：转换为FP16 GGUF格式"""
    print("\n🔄 步骤1: 转换为FP16 GGUF格式...")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fp16_gguf = Path(output_dir) / "model_fp16.gguf"
    
    try:
        cmd = [
            "python", str(convert_script),
            model_path,
            "--outfile", str(fp16_gguf),
            "--outtype", "f16"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=LLAMA_CPP_DIR)
        
        print(f"✅ FP16 GGUF转换完成: {fp16_gguf}")
        return fp16_gguf
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FP16转换失败: {e}")
        return None

def quantize_to_q4(fp16_gguf, output_dir, quantize_tool):
    """第二步：量化为Q4格式"""
    print("\n⚡ 步骤2: 量化为Q4格式...")
    
    q4_gguf = Path(output_dir) / "model_q4_0.gguf"
    
    try:
        cmd = [
            str(quantize_tool),
            str(fp16_gguf),
            str(q4_gguf),
            "Q4_0"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"✅ Q4量化完成: {q4_gguf}")
        
        # 清理中间文件
        if fp16_gguf.exists():
            fp16_gguf.unlink()
            print("🗑️ 清理中间FP16文件")
        
        return q4_gguf
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Q4量化失败: {e}")
        return None

def create_ollama_modelfile(gguf_path, model_name=MODEL_NAME):
    """创建Ollama Modelfile"""
    print(f"\n📋 创建Ollama Modelfile...")
    
    system_prompt = "You are a professional sociologist, proficient in simulation and prediction of social behavior. Meanwhile you are a helpful assistant."
    
    modelfile_content = f"""FROM {gguf_path}

# 系统提示
SYSTEM "{system_prompt}"

# Llama3.1 官方模板格式
TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

\"\"\"

# 生成参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""
    
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile创建完成: {modelfile_path}")
    return modelfile_path, model_name

def create_ollama_model(modelfile_path, model_name):
    """创建Ollama模型"""
    print(f"\n🚀 创建Ollama模型: {model_name}")
    
    try:
        # 检查ollama是否可用
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        
        # 创建模型
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"✅ Ollama模型创建成功: {model_name}")
        
        # 测试模型
        print("🧪 测试模型...")
        result = subprocess.run([
            "ollama", "run", model_name, "Hello, who are you?"
        ], capture_output=True, text=True, check=True)
        
        print(f"🤖 模型回答: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama操作失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 未找到ollama命令，请先安装Ollama")
        return False

def get_file_size(path):
    """获取文件大小"""
    if Path(path).exists():
        size_bytes = Path(path).stat().st_size
        size_gb = size_bytes / (1024**3)
        return f"{size_gb:.2f}GB"
    return "未知"

def main():
    """主函数"""
    print("🚀 HuggingFace模型转Q4量化工具")
    print("=" * 60)
    
    print(f"📋 配置信息:")
    print(f"   输入模型: {MERGED_MODEL_PATH}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   工具目录: {LLAMA_CPP_DIR}")
    print("=" * 60)
    
    # 检查输入模型
    if not Path(MERGED_MODEL_PATH).exists():
        print(f"❌ 输入模型不存在: {MERGED_MODEL_PATH}")
        return
    
    # 检查工具
    convert_script, quantize_tool = check_tools()
    if not convert_script:
        print("💥 缺少必要工具，无法继续")
        return
    
    # 步骤1: 转换为FP16 GGUF
    fp16_gguf = convert_to_fp16_gguf(MERGED_MODEL_PATH, OUTPUT_DIR, convert_script)
    if not fp16_gguf:
        print("💥 FP16转换失败")
        return
    
    # 步骤2: 量化为Q4
    if quantize_tool:
        q4_gguf = quantize_to_q4(fp16_gguf, OUTPUT_DIR, quantize_tool)
        if not q4_gguf:
            print("💥 Q4量化失败")
            return
    else:
        print("⚠️ 量化工具不可用，保留FP16格式")
        q4_gguf = fp16_gguf
    
    # 步骤3: 创建Ollama模型
    modelfile_path, model_name = create_ollama_modelfile(q4_gguf)
    success = create_ollama_model(modelfile_path, model_name)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 转换结果总结:")
    print(f"   原始模型大小: 约16GB (FP16)")
    print(f"   转换后大小: {get_file_size(q4_gguf)} (Q4_0)")
    print(f"   输出文件: {q4_gguf}")
    
    if success:
        print(f"✅ 转换和部署成功!")
        print(f"🎯 Ollama模型: {model_name}")
        print(f"🧪 测试命令: ollama run {model_name}")
    else:
        print("⚠️ 转换成功但Ollama部署失败")
        print(f"📁 手动部署: ollama create {model_name} -f {modelfile_path}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
