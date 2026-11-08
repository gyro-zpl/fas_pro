#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract loss data from training logs and plot loss curves (English version)
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file_path):
    """
    Parse training log file and extract loss and other training metrics
    """
    
    # Regex patterns for training log metrics
    loss_pattern = r"'loss': ([\d.]+)"
    epoch_pattern = r"'epoch': ([\d.]+)"
    lr_pattern = r"'learning_rate': ([\de.-]+)"
    grad_norm_pattern = r"'grad_norm': ([\d.]+)"
    
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    steps = []
    
    print(f"Reading log file: {log_file_path}")
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by lines and process each line
        lines = content.split('\n')
        step_count = 0
        
        for line in lines:
            # Skip empty lines and irrelevant lines
            if not line.strip() or 'loss' not in line:
                continue
                
            # Extract metrics
            loss_match = re.search(loss_pattern, line)
            epoch_match = re.search(epoch_pattern, line)
            lr_match = re.search(lr_pattern, line)
            grad_norm_match = re.search(grad_norm_pattern, line)
            
            if loss_match and epoch_match:
                loss = float(loss_match.group(1))
                epoch = float(epoch_match.group(1))
                
                losses.append(loss)
                epochs.append(epoch)
                steps.append(step_count)
                step_count += 1
                
                if lr_match:
                    learning_rates.append(float(lr_match.group(1)))
                else:
                    learning_rates.append(0.0)
                    
                if grad_norm_match:
                    grad_norms.append(float(grad_norm_match.group(1)))
                else:
                    grad_norms.append(0.0)
        
        print(f"Successfully extracted data from {len(losses)} training steps")
        
        return {
            'losses': losses,
            'epochs': epochs,
            'learning_rates': learning_rates,
            'grad_norms': grad_norms,
            'steps': steps
        }
        
    except Exception as e:
        print(f"Failed to read log file: {e}")
        return None

def plot_training_curves(data, output_dir="/home/cyyuan/FAS_PRO/sft"):
    """
    Plot training curves
    """
    
    if not data or len(data['losses']) == 0:
        print("No training data available")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['font.size'] = 10
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LoRA Fine-tuning Training Curves Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss curve (by steps)
    axes[0, 0].plot(data['steps'], data['losses'], 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Training Loss vs Steps', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics
    min_loss = min(data['losses'])
    final_loss = data['losses'][-1]
    axes[0, 0].axhline(y=min_loss, color='r', linestyle='--', alpha=0.7, label=f'Min Loss: {min_loss:.4f}')
    axes[0, 0].axhline(y=final_loss, color='g', linestyle='--', alpha=0.7, label=f'Final Loss: {final_loss:.4f}')
    axes[0, 0].legend()
    
    # 2. Loss curve (by epochs)
    axes[0, 1].plot(data['epochs'], data['losses'], 'purple', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Training Loss vs Epochs', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Training Epochs')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning rate curve
    if any(lr > 0 for lr in data['learning_rates']):
        axes[1, 0].plot(data['steps'], data['learning_rates'], 'orange', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    
    # 4. Gradient norm curve
    if any(gn > 0 for gn in data['grad_norms']):
        axes[1, 1].plot(data['steps'], data['grad_norms'], 'red', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Gradient Norm', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add average line
        avg_grad_norm = np.mean(data['grad_norms'])
        axes[1, 1].axhline(y=avg_grad_norm, color='orange', linestyle='--', alpha=0.7, 
                          label=f'Avg Grad Norm: {avg_grad_norm:.4f}')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No Gradient Norm Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Gradient Norm', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training curves saved to: {output_path}")
    
    # Also create a simple loss-only plot
    plt.figure(figsize=(12, 8))
    
    # Plot loss with smoothed trend
    plt.subplot(2, 1, 1)
    plt.plot(data['steps'], data['losses'], 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
    
    # Add smoothed trend line
    if len(data['losses']) > 10:
        window_size = max(5, len(data['losses']) // 20)
        smoothed_losses = []
        smoothed_steps = []
        for i in range(window_size, len(data['losses']) - window_size):
            smoothed_losses.append(np.mean(data['losses'][i-window_size:i+window_size]))
            smoothed_steps.append(data['steps'][i])
        
        plt.plot(smoothed_steps, smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    
    plt.title(f'Training Loss Progress (Total Steps: {len(data["losses"])}, Epochs: {data["epochs"][-1]:.1f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss by epochs
    plt.subplot(2, 1, 2)
    plt.plot(data['epochs'], data['losses'], 'g-', linewidth=1.5, alpha=0.8)
    plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save simple plot
    simple_output_path = Path(output_dir) / "loss_curve_simple.png"
    plt.savefig(simple_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simple loss curve saved to: {simple_output_path}")
    
    return output_path, simple_output_path

def print_training_summary(data):
    """
    Print training summary
    """
    
    if not data or len(data['losses']) == 0:
        print("No training data available")
        return
    
    losses = data['losses']
    epochs = data['epochs']
    
    print("\n" + "="*60)
    print("Training Summary Report")
    print("="*60)
    
    print(f"Training Configuration:")
    print(f"   Total Steps: {len(losses)}")
    print(f"   Total Epochs: {epochs[-1]:.2f}")
    print(f"   Initial Loss: {losses[0]:.4f}")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Best Loss: {min(losses):.4f}")
    print(f"   Loss Reduction: {losses[0] - losses[-1]:.4f} ({((losses[0] - losses[-1])/losses[0]*100):.2f}%)")
    
    # Calculate smoothed trend
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        smoothed_losses = []
        for i in range(len(losses) - window_size + 1):
            smoothed_losses.append(np.mean(losses[i:i+window_size]))
        
        trend = "Increasing" if smoothed_losses[-1] > smoothed_losses[0] else "Decreasing"
        print(f"   Loss Trend: {trend}")
    
    # Gradient norm statistics
    if data['grad_norms'] and any(gn > 0 for gn in data['grad_norms']):
        grad_norms = [gn for gn in data['grad_norms'] if gn > 0]
        print(f"\nGradient Statistics:")
        print(f"   Avg Gradient Norm: {np.mean(grad_norms):.4f}")
        print(f"   Max Gradient Norm: {max(grad_norms):.4f}")
        print(f"   Min Gradient Norm: {min(grad_norms):.4f}")
    
    # Learning rate statistics
    if data['learning_rates'] and any(lr > 0 for lr in data['learning_rates']):
        lr_values = [lr for lr in data['learning_rates'] if lr > 0]
        print(f"\nLearning Rate Statistics:")
        print(f"   Initial LR: {lr_values[0]:.2e}")
        print(f"   Final LR: {lr_values[-1]:.2e}")
    
    print("="*60)

def main():
    """Main function"""
    log_file_path = "/home/cyyuan/FAS_PRO/sft/nohup.out"
    output_dir = "/home/cyyuan/FAS_PRO/sft"
    
    print("LoRA Fine-tuning Training Curves Analysis Tool")
    print("="*60)
    
    # Parse log file
    data = parse_training_log(log_file_path)
    
    if data is None:
        print("Failed to parse log file")
        return
    
    # Print training summary
    print_training_summary(data)
    
    # Plot training curves
    print(f"\nGenerating training curves...")
    plot_paths = plot_training_curves(data, output_dir)
    
    print(f"\nAnalysis completed!")
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()

