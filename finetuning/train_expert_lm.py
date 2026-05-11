#!/usr/bin/env python3
"""Train a lightweight text-only expert causal LM from derived MAC-SLU corpora."""
import argparse
import json
import os


import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class TextOnlyExpertConfig(PretrainedConfig):
    model_type = "qwen3_asr_text_expert"

    def __init__(self, vocab_size, hidden_size, pad_token_id=None, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)


class TextOnlyExpertModel(PreTrainedModel):
    config_class = TextOnlyExpertConfig

    def __init__(self, config, text_model=None, lm_head=None):
        super().__init__(config)
        if text_model is None or lm_head is None:
            raise ValueError("text_model and lm_head are required")
        self.model = text_model
        self.lm_head = lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)


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
    ap.add_argument("--init_from_asr", action="store_true")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from qwen_asr import Qwen3ASRModel

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if args.init_from_asr:
        asr_wrapper = Qwen3ASRModel.from_pretrained(args.model_name_or_path)
        asr_model = asr_wrapper.model
        thinker = asr_model.thinker
        text_model = thinker.model
        lm_head = thinker.lm_head
        text_cfg = getattr(asr_model.config, "text_config", None)
        hidden_size = getattr(text_cfg, "hidden_size", lm_head.in_features)
        vocab_size = getattr(text_cfg, "vocab_size", lm_head.out_features)
        model = TextOnlyExpertModel(
            TextOnlyExpertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                pad_token_id=tokenizer.pad_token_id,
            ),
            text_model=text_model,
            lm_head=lm_head,
        )
    else:
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
