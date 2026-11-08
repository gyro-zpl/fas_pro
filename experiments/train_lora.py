"""LoRA fine-tuning for the PAS rating task (model-agnostic).

Uses the tokenizer's chat template, so it works for Qwen and Llama. Only the
answer tokens contribute to the loss.

Example:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.train_lora \
        --train-file data/anes/v3/splits/train_t.jsonl \
        --output-dir sft/sft_models/qwen3_anes_v3_seed42 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from experiments import config
from experiments.eval.readout import CANDIDATE_SYSTEM, BACKGROUND, GROUP_SYSTEM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RatingDataset(Dataset):
    """Pre-tokenizes every row once; per-step work is only padding."""

    def __init__(self, path: str, tokenizer, max_length: int = 768,
                 task: str = "candidate", few_shot: bool = False):
        self.rows = [json.loads(line) for line in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.few_shot = few_shot
        self.items = [self._encode(row) for row in self.rows]

    def _encode(self, row: dict) -> dict:
        system = CANDIDATE_SYSTEM if self.task == "candidate" else GROUP_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row["instruction"] + "\n" + BACKGROUND},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "qwen" in self.tokenizer.name_or_path.lower():
            kwargs["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        answer = str(int(round(float(row["output"])))) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = len(features)
        input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((batch, max_len), dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"])
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(f["labels"])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default=str(config.V3_SPLITS / "train_t.jsonl"))
    p.add_argument("--eval-file", default=None)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--task", default="candidate", choices=["candidate", "group"])
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="lower memory, ~2-3x slower")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = config.model_info(args.model)
    print(f"model={info['key']} path={info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation=args.attn_implementation,
    )
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    from peft import LoraConfig, TaskType, get_peft_model
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = RatingDataset(args.train_file, tokenizer, args.max_length, args.task)
    if args.max_samples:
        train_ds.items = train_ds.items[: args.max_samples]
    eval_ds = None
    if args.eval_file:
        eval_ds = RatingDataset(args.eval_file, tokenizer, args.max_length, args.task)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
