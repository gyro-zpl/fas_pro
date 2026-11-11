"""Model loading, chat prompting, and rating readout.

Tokenization note
-----------------
Qwen3's tokenizer splits 2- and 3-digit numbers into one token per digit, so a
single-position softmax over 0-100 is not possible. We therefore use constrained
greedy generation of a short number string and parse it. `inspect_rating_tokens`
reports the tokenization so the choice is auditable.

For Llama (if weights become available), the same code path works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import config

CANDIDATE_SYSTEM = (
    "You are an expert in political science specializing in voter behavior prediction. "
    "Your task is to predict a voter's support score (0-100) for a political candidate "
    "based on the voter's demographics, political views, and the candidate's characteristics. "
    "Respond with ONLY a number between 0 and 100."
)

GROUP_SYSTEM = (
    "You are an expert in political science and social attitudes. "
    "Your task is to predict a voter's feeling thermometer score (0-100) toward a social group "
    "based on the voter's political identity, ideology, and age. "
    "Respond with ONLY a number between 0 and 100."
)

BACKGROUND = (
    "\nThe data comes from American National Election Studies (ANES). "
    "Consider the political context of the year and general approval ratings."
)

# Five anchors spanning the score range. Kept short and number-only so parsing is
# unambiguous; this differs from the legacy prompt, which included reasoning text.
FEW_SHOT = """Here are 5 examples covering different score ranges:

Voter: 38-year-old Female White, Some high school; Strong Republican, Conservative
Candidate: Trump (Republican, President), Year: 2024
Predicted score: 92

Voter: 38-year-old Male White, Some high school; Strong Republican, Moderate
Candidate: Yang (Democrat, Candidate), Year: 2020
Predicted score: 60

Voter: 43-year-old Male, Some high school; Independent, Moderate
Candidate: Buttigieg (Democrat, Candidate), Year: 2020
Predicted score: 40

Voter: 38-year-old Female Hispanic, Some high school; Lean Democrat, Liberal
Candidate: Trump (Republican, President), Year: 2024
Predicted score: 20

Voter: 62-year-old Male White, Some college; Strong Democrat, Liberal
Candidate: Trump (Republican, President), Year: 2024
Predicted score: 2

Respond with only the number.
"""


def parse_rating(text: str) -> int | None:
    """Extract the first integer in [0, 100] from model output."""
    if text is None:
        return None
    # If the model emitted a thinking block, only score the final answer.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace(",", "")
    for m in re.finditer(r"\d{1,3}(?:\.\d+)?", text):
        v = float(m.group(0))
        if v.is_integer() and 0 <= v <= 100:
            return int(v)
    return None


def inspect_rating_tokens(tokenizer) -> dict:
    single, multi = [], []
    for n in range(101):
        ids = tokenizer(str(n), add_special_tokens=False).input_ids
        (single if len(ids) == 1 else multi).append((n, ids))
    return {
        "n_single": len(single),
        "n_multi": len(multi),
        "multi_examples": multi[:10],
    }


def build_messages(
    instruction: str,
    few_shot: bool = False,
    task: str = "candidate",
    extra_system: str | None = None,
) -> list[dict]:
    system = CANDIDATE_SYSTEM if task == "candidate" else GROUP_SYSTEM
    if extra_system:
        system = f"{system}\n\n{extra_system}"
    if few_shot:
        system = f"{system}\n\n{FEW_SHOT}"
    user = instruction + "\n" + BACKGROUND
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def apply_chat(tokenizer, messages: list[dict], family: str,
               enable_thinking: bool | None = None) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if family == "qwen":
        kwargs["enable_thinking"] = bool(enable_thinking) if enable_thinking is not None else False
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    # For DeepSeek-R1-Distill the template ends with "<think>\n"; the model is
    # fine-tuned to emit "</think>\n<score>" immediately. Do not pre-close the
    # block here: the base model ignores a pre-closed block and still reasons.
    return prompt


@dataclass
class RatingGenerator:
    model: object
    tokenizer: object
    family: str = "qwen"
    batch_size: int = 8
    max_new_tokens: int = 8

    @classmethod
    def load(cls, model_key: str | None = None, lora_path: str | None = None,
             batch_size: int = 8) -> "RatingGenerator":
        info = config.model_info(model_key)
        tokenizer = AutoTokenizer.from_pretrained(info["path"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()

        return cls(model=model, tokenizer=tokenizer, family=info["family"],
                   batch_size=batch_size)

    @torch.no_grad()
    def predict(
        self,
        instructions: Iterable[str],
        few_shot: bool = False,
        task: str = "candidate",
        enable_thinking: bool | None = None,
        extra_system: str | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        instructions = list(instructions)
        results: list[dict] = []
        for start in range(0, len(instructions), self.batch_size):
            batch = instructions[start:start + self.batch_size]
            prompts = [
                apply_chat(
                    self.tokenizer,
                    build_messages(x, few_shot, task, extra_system),
                    self.family,
                    enable_thinking=enable_thinking,
                )
                for x in batch
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=1024,
            ).to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text in texts:
                results.append({"text": text, "pred": parse_rating(text)})
            if (start // self.batch_size) % 10 == 0:
                print(f"[predict] {min(start + len(batch), len(instructions))}/{len(instructions)}",
                      flush=True)
        return results
