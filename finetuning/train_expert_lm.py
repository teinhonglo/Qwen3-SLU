#!/usr/bin/env python3
"""Train a lightweight text-only expert causal LM from derived MAC-SLU corpora."""
import argparse
import json
import os



def load_rows(path):
    """Load JSONL rows from path with explicit error if file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input jsonl not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_id}: {exc}") from exc
    return rows


def main():
    """CLI entry for expert LM training."""
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
        return Dataset.from_list(
            [{"text": f"{r.get('input_text', '')}\n{r.get('target_text', '')}"} for r in rows]
        )

    train_ds = to_dataset(load_rows(args.train_jsonl))
    dev_ds = to_dataset(load_rows(args.dev_jsonl))

    def preprocess(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(preprocess, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
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
