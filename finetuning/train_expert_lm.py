#!/usr/bin/env python3
"""Train a lightweight text-only expert causal LM from prefix->next-token corpora."""
import argparse
import json
import os


def load_rows(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input jsonl not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_id}: {exc}") from exc
            if "prefix_text" not in row or "target_token_text" not in row:
                raise ValueError(f"Missing prefix_text/target_token_text in {path}:{line_id}")
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser("Train lightweight expert causal LM")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--dev_jsonl", required=True)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    def to_dataset(rows):
        return Dataset.from_list(rows)

    def preprocess(example):
        prefix = example["prefix_text"]
        target = example["target_token_text"]
        full = prefix + target
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target_ids = full_ids[len(prefix_ids):]

        full_ids = full_ids[: args.max_length]
        attn = [1] * len(full_ids)
        labels = [-100] * len(full_ids)

        start = min(len(prefix_ids), len(full_ids))
        max_tgt = max(0, len(full_ids) - start)
        for i, tid in enumerate(target_ids[:max_tgt]):
            labels[start + i] = tid

        pad_len = args.max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}

    train_ds = to_dataset(load_rows(args.train_jsonl)).map(preprocess, remove_columns=to_dataset(load_rows(args.train_jsonl)).column_names)
    dev_ds = to_dataset(load_rows(args.dev_jsonl)).map(preprocess, remove_columns=to_dataset(load_rows(args.dev_jsonl)).column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
