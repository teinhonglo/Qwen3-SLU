#!/usr/bin/env python3
"""Train a lightweight text-only expert causal LM from derived MAC-SLU corpora."""
import argparse
import json
import os


import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class TextOnlyExpertConfig(PretrainedConfig):
    model_type = "qwen3_asr_text_expert"

    def __init__(self, vocab_size, hidden_size, pad_token_id=None, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)


class TextOnlyExpertModel(PreTrainedModel, GenerationMixin):
    config_class = TextOnlyExpertConfig
    supports_gradient_checkpointing = True

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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            if "input_ids" not in model_inputs:
                model_inputs["input_ids"] = input_ids
            if attention_mask is not None and "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = attention_mask
            return model_inputs
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        model_inputs.update(kwargs)
        return model_inputs

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


def load_train_conf(train_conf_path: str):
    if not train_conf_path:
        return None

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf must be a list in format: [training_args, model_args]")

    training_args, model_args = cfg
    if not isinstance(training_args, dict) or not isinstance(model_args, dict):
        raise ValueError("train_conf entries must both be dictionaries")
    return [training_args, model_args]


def build_single_step_examples(rows, tokenizer, max_length):
    """Expand each row into multiple samples, each supervising one next token.

    For each row, we tokenize step-wise contexts that simulate autoregressive
    decoding: prefix + target[:k].

    Each constructed sample supervises exactly one next token (the last token
    in the step-wise context), and all other label positions are set to -100.
    """
    examples = []
    for row in rows:
        prefix = str(row.get("input_text", ""))
        target = str(row.get("target_text", ""))
        if not target:
            continue

        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )["input_ids"]

        # Build true single-step contexts: prefix + partial_target.
        # For step t, the context includes target tokens up to t, and we only
        # supervise the token at the last non-padding position.
        for step_end in range(1, len(target) + 1):
            merged = f"{prefix}{target[:step_end]}"
            encoded = tokenizer(
                merged,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            seq_len = int(sum(attention_mask))
            if seq_len <= 0:
                continue

            prefix_len = min(len(prefix_ids), seq_len)
            label_pos = seq_len - 1
            if label_pos < prefix_len:
                continue

            labels = [-100] * len(input_ids)
            labels[label_pos] = input_ids[label_pos]
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    return examples


def main():
    """CLI entry for expert LM training."""
    ap = argparse.ArgumentParser("Train lightweight expert causal LM")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--dev_jsonl", required=True)
    ap.add_argument("--model_name_or_path", default=None, help="Optional override. If empty, use model_args.model_path from train_conf")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_conf", required=True, help="JSON config: [training_args, model_args]")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_asr import Qwen3ASRModel

    train_conf = load_train_conf(args.train_conf)
    if train_conf is None:
        raise ValueError("--train_conf is required")

    training_args_conf, model_args_conf = train_conf
    training_args_conf = dict(training_args_conf)
    model_args_conf = dict(model_args_conf)

    model_name_or_path = args.model_name_or_path or model_args_conf.get("model_path")
    if not model_name_or_path:
        raise ValueError("model path is required: set --model_name_or_path or model_args.model_path in train_conf")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if model_args_conf.get("init_from_asr", True):
        asr_wrapper = Qwen3ASRModel.from_pretrained(model_name_or_path)
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
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    lora_type = model_args_conf.get("lora_type", "default")
    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        if lora_type != "default":
            raise ValueError(f"Unsupported lora_type for train_expert_lm.py: {lora_type}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_config,
        )
        peft_config.base_model_name_or_path = model_name_or_path
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    max_length = int(model_args_conf.get("max_length", 256))
    train_rows = load_rows(args.train_jsonl)
    dev_rows = load_rows(args.dev_jsonl)
    train_examples = build_single_step_examples(train_rows, tokenizer, max_length)
    dev_examples = build_single_step_examples(dev_rows, tokenizer, max_length)
    if not train_examples:
        raise ValueError("No train examples created for single-step next-token training")
    if not dev_examples:
        raise ValueError("No dev examples created for single-step next-token training")

    print(f"[info] single-step train examples: {len(train_examples)} from {len(train_rows)} rows")
    print(f"[info] single-step dev examples: {len(dev_examples)} from {len(dev_rows)} rows")

    train_ds = Dataset.from_list(train_examples)
    dev_ds = Dataset.from_list(dev_examples)

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = has_cuda and not use_bf16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        bf16=use_bf16,
        fp16=use_fp16,
        **training_args_conf,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if train_conf is not None and trainer.args.process_index == 0:
        saved_train_conf = os.path.join(args.output_dir, "train_conf.json")
        with open(saved_train_conf, "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
