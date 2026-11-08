#!/usr/bin/env python3
"""快速查看训练日志"""
import json
import sys
from pathlib import Path

def view_training_log(checkpoint_path):
    """查看训练日志"""
    trainer_state_file = Path(checkpoint_path) / "trainer_state.json"
    
    if not trainer_state_file.exists():
        print(f"❌ 找不到日志文件: {trainer_state_file}")
        return
    
    with open(trainer_state_file, 'r') as f:
        state = json.load(f)
    
    print("="*80)
    print("训练状态概览")
    print("="*80)
    print(f"当前Step: {state['global_step']}")
    print(f"当前Epoch: {state['epoch']:.2f}")
    print(f"最佳Step: {state['best_global_step']}")
    print(f"最佳Eval Loss: {state['best_metric']:.4f}")
    print(f"最佳Checkpoint: {state['best_model_checkpoint']}")
    
    print("\n" + "="*80)
    print("训练历史（最近20条）")
    print("="*80)
    
    log_history = state['log_history']
    for entry in log_history[-20:]:
        if 'loss' in entry and 'eval_loss' not in entry:
            # 训练loss
            print(f"Step {entry['step']:4d} | Epoch {entry['epoch']:.2f} | "
                  f"Loss: {entry['loss']:.4f} | Grad: {entry['grad_norm']:.4f}")
        elif 'eval_loss' in entry:
            # 评估loss
            print(f"Step {entry['step']:4d} | Epoch {entry['epoch']:.2f} | "
                  f"Eval Loss: {entry['eval_loss']:.4f} ⭐")
    
    # 分析loss趋势
    print("\n" + "="*80)
    print("Loss趋势分析")
    print("="*80)
    
    train_losses = [e['loss'] for e in log_history if 'loss' in e and 'eval_loss' not in e]
    eval_losses = [e['eval_loss'] for e in log_history if 'eval_loss' in e]
    
    if train_losses:
        print(f"训练Loss:")
        print(f"  初始: {train_losses[0]:.4f}")
        print(f"  当前: {train_losses[-1]:.4f}")
        print(f"  最低: {min(train_losses):.4f}")
        print(f"  下降: {train_losses[0] - train_losses[-1]:.4f} ({(train_losses[0]-train_losses[-1])/train_losses[0]*100:.1f}%)")
    
    if eval_losses:
        print(f"\n评估Loss:")
        print(f"  初始: {eval_losses[0]:.4f}")
        print(f"  当前: {eval_losses[-1]:.4f}")
        print(f"  最低: {min(eval_losses):.4f}")
        print(f"  下降: {eval_losses[0] - eval_losses[-1]:.4f} ({(eval_losses[0]-eval_losses[-1])/eval_losses[0]*100:.1f}%)")
    
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "/home/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora/checkpoint-400"
    
    view_training_log(checkpoint_path)
