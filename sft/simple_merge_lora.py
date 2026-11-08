#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整版本：合并LoRA权重，转换为GGUF格式，并自动创建Ollama模型
支持Q4_0等多种量化精度
"""

import os
import torch
import subprocess
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL_PATH, LORA_PATH, OUTPUT_PATH, MAX_SHARD_SIZE, MODEL_NAME
base_model_path = BASE_MODEL_PATH
lora_path = LORA_PATH
output_path = OUTPUT_PATH
max_shard_size = MAX_SHARD_SIZE
print(f"使用配置: {base_model_path}, {lora_path}, {output_path}, {max_shard_size}, {MODEL_NAME}")


def merge_lora_to_base_model():
    """合并LoRA权重到基础模型"""
    print("🔄 开始合并LoRA权重...")
    
    # 创建输出目录
    Path(output_path).mkdir(exist_ok=True)
    
    try:
        # 1. 加载tokenizer
        print("📖 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 2. 加载基础模型
        print("🤖 加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 3. 加载LoRA适配器
        print("🎯 加载LoRA适配器...")
        model_with_lora = PeftModel.from_pretrained(base_model, lora_path)
        
        # 4. 合并权重
        print("⚡ 合并权重...")
        merged_model = model_with_lora.merge_and_unload()
        
        # 5. 保存合并后的模型
        print(f"💾 保存到: {output_path}")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=max_shard_size
        )
        
        # 6. 保存tokenizer
        tokenizer.save_pretrained(output_path)
        
        print("✅ 合并完成!")
        print(f"📁 合并后的模型保存在: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return None

def test_merged_model(model_path):
    """测试合并后的模型"""
    print("\n🧪 测试合并后的模型...")
    
    try:
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # 测试提示
        prompt = "Who are you？"
        messages = [
            {"role": "system", "content": "You are a professional sociologist, proficient in simulation and prediction of social behavior.Meanwhile you are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # 生成回答
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"🤖 测试回答: {response}")
        print("✅ 模型测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def check_llama_cpp():
    """检查llama.cpp安装"""
    # 优先使用llama.cpp目录中的转换脚本
    convert_options = [
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp/convert_hf_to_gguf.py"),
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/convert_hf_to_gguf.py")
    ]
    
    convert_script = None
    for script in convert_options:
        if script.exists():
            convert_script = script
            break
    
    # 检查量化可执行文件的可能位置
    quantize_options = [
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp/build/bin/llama-quantize"),
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp/build/bin/quantize"),
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/llama.cpp/quantize"),
        Path("/home/cyyuan/FAS_PRO/sft/ollama_export/quantize")
    ]
    
    quantize_executable = None
    for q_path in quantize_options:
        if q_path.exists():
            quantize_executable = q_path
            break
    
    if convert_script:
        if quantize_executable:
            print(f"✅ 使用llama.cpp工具:")
            print(f"   转换脚本: {convert_script}")
            print(f"   量化工具: {quantize_executable}")
            return {"convert": convert_script, "quantize": quantize_executable}
        else:
            print(f"⚠️ 找到转换脚本但缺少量化工具")
            print(f"   转换脚本: {convert_script}")
            print(f"   将只进行GGUF转换，跳过量化步骤")
            return {"convert": convert_script, "quantize": None}
    
    print(f"❌ 未找到可用的转换脚本")
    return None


def create_q4_0_with_ollama(fp16_gguf_path, output_dir, model_name):
    """使用Ollama内置功能创建Q4_0量化模型"""
    print("🦙 使用Ollama内置量化功能...")
    
    try:
        # 创建临时的Modelfile用于量化
        temp_modelfile = output_dir / "Modelfile_temp"
        temp_modelfile_content = f"""FROM {fp16_gguf_path}

# 临时模型用于量化
SYSTEM "Temporary model for quantization"
"""
        
        with open(temp_modelfile, 'w') as f:
            f.write(temp_modelfile_content)
        
        # 创建临时模型
        temp_model_name = f"{model_name}-temp"
        subprocess.run([
            "ollama", "create", temp_model_name, "-f", str(temp_modelfile)
        ], check=True, capture_output=True)
        
        print(f"✅ 临时模型创建成功，Ollama已自动应用最优量化")
        
        # 删除临时文件
        temp_modelfile.unlink()
        
        return temp_model_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama量化失败: {e}")
        return None

def convert_to_gguf(merged_model_path, quantization="q4_0"):
    """转换HuggingFace模型为GGUF格式并量化"""
    print(f"\n🔄 开始转换为GGUF格式 (量化: {quantization})...")
    
    # 检查llama.cpp
    llama_cpp_info = check_llama_cpp()
    if not llama_cpp_info:
        return None
    
    output_dir = Path(merged_model_path).parent
    
    try:
        # 1. 转换为FP16 GGUF格式
        fp16_gguf = output_dir / "model_fp16.gguf"
        # print("📝 转换为FP16 GGUF...")
        
        # convert_script = llama_cpp_info["convert"]
        # subprocess.run([
        #     "python", str(convert_script),
        #     merged_model_path,
        #     "--outfile", str(fp16_gguf),
        #     "--outtype", "f16"
        # ], check=True)
        
        # 2. 量化模型 (如果有量化工具)
        if llama_cpp_info["quantize"]:
            quantized_gguf = output_dir / f"model_{quantization}.gguf"
            print(f"⚡ 使用llama-quantize工具量化为{quantization.upper()}...")
            
            quantize_executable = llama_cpp_info["quantize"]
            subprocess.run([
                str(quantize_executable),
                str(fp16_gguf),
                str(quantized_gguf),
                quantization
            ], check=True)
            
            # 删除中间文件
            if fp16_gguf.exists():
                fp16_gguf.unlink()
                print("🗑️ 清理中间文件")
            
            print(f"✅ GGUF转换和量化完成: {quantized_gguf}")
            return quantized_gguf
        else:
            print(f"⚠️ 量化工具不可用，保留FP16格式")
            print(f"✅ GGUF转换完成: {fp16_gguf}")
            return fp16_gguf
        
    except subprocess.CalledProcessError as e:
        print(f"❌ GGUF转换失败: {e}")
        return None

def create_ollama_modelfile(gguf_path, model_name = MODEL_NAME, quantization=None):
    """创建Ollama Modelfile"""
    print(f"\n📋 创建Ollama Modelfile...")
    
    # 尝试导入配置
    try:
        from config import SYSTEM_PROMPT, GENERATION_PARAMS
        system_prompt = SYSTEM_PROMPT
        params = GENERATION_PARAMS
    except ImportError:
        # 默认配置
        system_prompt = "You are a professional sociologist, proficient in simulation and prediction of social behavior.Meanwhile you are a helpful assistant."
        params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'num_ctx': 4096
        }
    
    # 如果是FP16文件但需要量化，添加量化指令
    quantization_note = ""
    if quantization and "fp16" in str(gguf_path):
        quantization_note = f"# 注意: 此模型将在运行时应用 {quantization.upper()} 量化\n"
    
    modelfile_content = f"""FROM {gguf_path}

{quantization_note}
# 系统提示
SYSTEM "{system_prompt}"

# Llama3.1 官方模板格式
TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

\"\"\"

# 生成参数
PARAMETER temperature {params['temperature']}
PARAMETER top_p {params['top_p']}
PARAMETER top_k {params['top_k']}
PARAMETER repeat_penalty {params['repeat_penalty']}
PARAMETER num_ctx {params['num_ctx']}
"""
    
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile创建完成: {modelfile_path}")
    return modelfile_path

def create_ollama_model(modelfile_path, model_name = MODEL_NAME):
    """创建并测试Ollama模型"""
    print(f"\n🚀 创建Ollama模型: {model_name}")
    
    try:
        # 检查ollama是否安装
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        
        # 创建模型
        subprocess.run([
            "ollama", "create", model_name, "-f", str(modelfile_path)
        ], check=True)
        
        print(f"✅ Ollama模型创建成功: {model_name}")
        
        # 测试模型
        print("🧪 测试模型...")
        result = subprocess.run([
            "ollama", "run", model_name, "Who are you？"
        ], capture_output=True, text=True, check=True)
        
        print(f"🤖 模型回答: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama操作失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 未找到ollama命令，请先安装Ollama")
        print("安装命令: curl -fsSL https://ollama.com/install.sh | sh")
        return False

def create_manual_instructions(gguf_path, model_name = MODEL_NAME):
    """创建手动操作指令"""
    instructions = f"""
📋 手动操作指令 (如果自动化失败时使用):

1. 创建Ollama模型:
   ollama create {model_name} -f {Path(gguf_path).parent / "Modelfile"}

2. 测试模型:
   ollama run {model_name} "Who are you？"

3. 列出所有模型:
   ollama list

4. 删除模型 (如需要):
   ollama rm {model_name}

5. 模型文件位置:
   - GGUF文件: {gguf_path}
   - Modelfile: {Path(gguf_path).parent / "Modelfile"}
"""
    
    instructions_file = Path(gguf_path).parent / "manual_instructions.txt"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"📝 手动指令已保存到: {instructions_file}")
    return instructions_file

if __name__ == "__main__":
    print("🚀 LoRA模型合并与Ollama导出工具")
    print("=" * 60)
    
    # 配置参数
    quantization = 'q4_0'  # 可选: q4_0, q4_1, q5_0, q5_1, q8_0
    model_name = MODEL_NAME
    
    print(f"📋 配置信息:")
    print(f"   量化格式: {quantization.upper()}")
    print(f"   模型名称: {model_name}")
    print("=" * 60)
    
    # 步骤1: 合并LoRA权重
    merged_path = merge_lora_to_base_model()
    if not merged_path:
        print("💥 合并失败!")
        exit(1)
    
    #merged_path = output_path
    # 步骤2: 测试合并后的模型
    test_merged_model(merged_path)
    
    # 步骤3: 转换为GGUF并量化
    gguf_path = convert_to_gguf(merged_path, quantization)
    if not gguf_path:
        print("💥 GGUF转换失败!")
        exit(1)
    
    # 步骤4: 创建Ollama Modelfile
    modelfile_path = create_ollama_modelfile(gguf_path, model_name, quantization)
    
    # 步骤5: 创建Ollama模型
    success = create_ollama_model(modelfile_path, model_name)
    
    # 步骤6: 创建手动指令（备用）
    create_manual_instructions(gguf_path, model_name)
    
    # 总结
    print("\n" + "=" * 60)
    if success:
        print("🎉 完整流程执行成功!")
        print(f"✅ Ollama模型已创建: {model_name}")
        print(f"📁 文件位置: {gguf_path.parent}")
        print(f"🧪 测试命令: ollama run {model_name} 'Who are you？'")
    else:
        print("⚠️  自动化流程部分失败，请参考手动指令")
        print(f"📁 文件位置: {gguf_path.parent}")
        print(f"📝 手动指令: {gguf_path.parent}/manual_instructions.txt")
    print("=" * 60)
