#!/usr/bin/env python3
"""
sft_train_lora.py — LoRA fine-tuning on Qwen2.5-7B-Instruct for NYC Metro.

Adapted from DTBK sft_02_train_lora.py. Key changes:
- Model: Qwen2.5-7B-Instruct (not Llama)
- Chat template: uses tokenizer.apply_chat_template()
- Larger training set (500k rows vs 35k)
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)

BASE = "/scratch/sx2490/econai/nyc_metro"
HF_CACHE = "/scratch/sx2490/hf_cache"


def build_chat_qwen(system_msg, user_msg, assistant_msg, tokenizer):
    """Build Qwen chat format using tokenizer's template."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def preprocess(example, tokenizer, max_length):
    """Tokenize and mask loss to assistant response only."""
    full_text = build_chat_qwen(
        example["system"], example["user"], example["assistant"], tokenizer
    )

    # Tokenize full conversation
    enc = tokenizer(full_text, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    # Find where assistant response starts
    # Tokenize everything up to assistant response
    prompt_messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length,
                           return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Labels: -100 for prompt tokens (no loss), actual ids for assistant response
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    # Also mask padding
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_jsonl", default=f"{BASE}/train.jsonl")
    parser.add_argument("--val_jsonl", default=f"{BASE}/val.jsonl")
    parser.add_argument("--output_dir", default=f"{BASE}/lora_qwen_nyc")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory, or 'auto' to use latest")
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help="Evaluation frequency in steps (default 2000, was 500 in v1)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Checkpoint save frequency in steps")
    parser.add_argument("--max_eval_samples", type=int, default=2000,
                        help="Cap on val set size to speed up eval (default 2000)")
    args = parser.parse_args()

    print(f"=== LoRA Training: Qwen2.5-7B-Instruct ===")
    print(f"  Model: {args.model_path}")
    print(f"  Train: {args.train_jsonl}")
    print(f"  Output: {args.output_dir}")

    # Load tokenizer (local path, no cache_dir needed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": args.train_jsonl,
        "validation": args.val_jsonl,
    })
    print(f"  Train: {len(dataset['train']):,}, Val: {len(dataset['validation']):,}")

    # Preprocess
    def preprocess_fn(example):
        return preprocess(example, tokenizer, args.max_length)

    print("Tokenizing...")
    tokenized = dataset.map(preprocess_fn, remove_columns=dataset["train"].column_names,
                            num_proc=4, load_from_cache_file=False)

    # Cap eval set size to speed up eval
    if args.max_eval_samples > 0 and len(tokenized["validation"]) > args.max_eval_samples:
        tokenized["validation"] = tokenized["validation"].select(range(args.max_eval_samples))
        print(f"  Capped val set to {args.max_eval_samples} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=42,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    # Resume from checkpoint if specified
    resume = args.resume_from_checkpoint
    if resume == "auto":
        import glob
        ckpts = sorted(glob.glob(f"{args.output_dir}/checkpoint-*"),
                       key=lambda p: int(p.split("-")[-1]))
        resume = ckpts[-1] if ckpts else None
        if resume:
            print(f"Auto-detected checkpoint: {resume}")

    print("Starting training...")
    if resume:
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # Save final LoRA weights
    final_path = f"{args.output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved LoRA weights to {final_path}")


if __name__ == "__main__":
    main()
