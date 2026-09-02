#!/usr/bin/env python3
"""Two-optimizer RobustGER training loop for Qwen3 on a single RTX 3090.

The loop follows Algorithm 1 from RobustGER:
1. update MINE on language-noise versus clean/noisy audio pairs;
2. update only the RobustGER adapter/tuner with H2T CE minus the joint MI term.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

from finetuning.qwen3_robustger_model import RobustGERForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RobustGERDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows found in {self.manifest_path}")
        self.language_noise = np.load(self.root / "language_noise.npy", mmap_mode="r")
        self.noisy_audio = np.load(self.root / "noisy_audio.npy", mmap_mode="r")
        self.clean_audio = np.load(self.root / "clean_audio.npy", mmap_mode="r")
        if len(self.rows) != self.language_noise.shape[0]:
            raise ValueError(f"Manifest/cache length mismatch in {self.root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        cache_index = int(row["index"])
        return {
            "input_text": row["input_text"],
            "target_text": row["target_text"],
            "language_noise": np.asarray(self.language_noise[cache_index], dtype=np.float32),
            "noisy_audio": np.asarray(self.noisy_audio[cache_index], dtype=np.float32),
            "clean_audio": np.asarray(self.clean_audio[cache_index], dtype=np.float32),
        }


class H2TCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        labels = []
        for item in items:
            prompt_ids = self.tokenizer(
                item["input_text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            target_ids = self.tokenizer(
                item["target_text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if self.eos_id is not None:
                target_ids = target_ids + [self.eos_id]

            # Preserve the complete target and keep the latest part of the
            # N-best prompt only if a sample exceeds the model context.
            if len(target_ids) >= self.max_length:
                target_ids = target_ids[: self.max_length - 1]
            prompt_budget = max(self.max_length - len(target_ids), 0)
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            ids = prompt_ids + target_ids
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            labels.append(
                torch.tensor(
                    [-100] * len(prompt_ids) + target_ids,
                    dtype=torch.long,
                )
            )

        max_len = max(len(ids) for ids in input_ids)
        batch_input = torch.full((len(items), max_len), self.pad_id, dtype=torch.long)
        batch_labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        attention = torch.zeros((len(items), max_len), dtype=torch.long)
        for index, (ids, target) in enumerate(zip(input_ids, labels)):
            batch_input[index, : len(ids)] = ids
            batch_labels[index, : len(target)] = target
            attention[index, : len(ids)] = 1

        return {
            "input_ids": batch_input,
            "attention_mask": attention,
            "labels": batch_labels,
            "language_noise": torch.from_numpy(
                np.stack([item["language_noise"] for item in items])
            ),
            "noisy_audio": torch.from_numpy(
                np.stack([item["noisy_audio"] for item in items])
            ),
            "clean_audio": torch.from_numpy(
                np.stack([item["clean_audio"] for item in items])
            ),
        }


class MINE(nn.Module):
    """MINE statistic network with raw-E_LN projection for stage 1."""

    def __init__(self, hidden_dim: int, audio_dim: int, raw_noise_dim: int):
        super().__init__()
        mine_hidden = max(hidden_dim // 4, 128)
        self.raw_prefix = nn.Linear(raw_noise_dim, hidden_dim, bias=False)
        self.language = nn.Linear(hidden_dim, mine_hidden, bias=False)
        self.audio = nn.Linear(audio_dim, mine_hidden, bias=False)
        self.combine = nn.Linear(mine_hidden, max(mine_hidden // 4, 32), bias=False)
        self.score = nn.Linear(max(mine_hidden // 4, 32), 1, bias=False)

    def forward(self, language: torch.Tensor, audio: torch.Tensor, raw: bool) -> torch.Tensor:
        language = language.float()
        audio = audio.float()
        if raw:
            language = self.raw_prefix(language)
        if language.ndim == 3:
            language = language.mean(dim=1)
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        language = F.silu(self.language(language))
        audio = F.silu(self.audio(audio))
        hidden = F.silu(self.combine(language + audio))
        return self.score(hidden).squeeze(-1)


def mine_bound(
    mine: MINE,
    language_noise: torch.Tensor,
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
) -> torch.Tensor:
    """Donsker-Varadhan joint-minus-marginal estimate used in stage 1."""
    joint = mine(-language_noise, noisy_audio, raw=True)
    marginal = mine(-language_noise, clean_audio, raw=True)
    return joint.mean() - (torch.logsumexp(marginal, dim=0) - math.log(marginal.numel()))


def build_model(config: Dict[str, Any], device: torch.device) -> RobustGERForCausalLM:
    base_config = AutoConfig.from_pretrained(config["base_model"])
    base_config.robustger_noise_dim = int(config["noise_dim"])
    base_config.robustger_adapter_prompt_length = int(config["adapter_prompt_length"])
    base_config.robustger_adapter_start_layer = int(config["adapter_start_layer"])
    base_config._attn_implementation = "eager"
    dtype = torch.float16 if device.type == "cuda" and config["dtype"] == "float16" else torch.float32
    model = RobustGERForCausalLM.from_pretrained(
        config["base_model"],
        config=base_config,
        torch_dtype=dtype,
    ).to(device)
    model.freeze_base_parameters()
    return model


def cross_entropy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(
    model: RobustGERForCausalLM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        noise = batch["language_noise"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                input_ids=inputs,
                attention_mask=attention,
                noise_embedding=noise,
                use_cache=False,
            )
        losses.append(cross_entropy_from_logits(outputs.logits, labels).item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: RobustGERForCausalLM,
    mine: MINE,
    config: Dict[str, Any],
    step: int,
    eval_loss: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.adapter_state_dict(),
            "mine": mine.state_dict(),
            "step": step,
            "eval_loss": eval_loss,
        },
        output_dir / name,
    )
    with (output_dir / "model_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model": config["base_model"],
                "n_best": config["n_best"],
                "language_noise_slots": config["n_best"] * (config["n_best"] - 1),
                "noise_dim": config["noise_dim"],
                "adapter_prompt_length": config["adapter_prompt_length"],
                "adapter_start_layer": config["adapter_start_layer"],
                "eval_loss": eval_loss,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RobustGER for Qwen3")
    parser.add_argument("--train_conf", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.train_conf, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    seed = int(config["seed"] if args.seed is None else args.seed)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_conf.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = H2TCollator(tokenizer, int(config["max_length"]))
    train_data = RobustGERDataset(args.train_file)
    eval_data = RobustGERDataset(args.eval_file)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = build_model(config, device)
    hidden_dim = int(model.config.hidden_size)
    mine = MINE(
        hidden_dim=hidden_dim,
        audio_dim=int(config["audio_dim"]),
        raw_noise_dim=int(config["noise_dim"]),
    ).to(device)
    model_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        model_params,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer_m = torch.optim.AdamW(
        mine.parameters(),
        lr=float(config["mine_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_eval = float("inf")
    global_step = 0
    grad_accumulation = int(config["gradient_accumulation_steps"])

    for epoch in range(int(config["epochs"])):
        model.train()
        mine.train()
        for batch_index, batch in enumerate(train_loader):
            language_noise = batch["language_noise"].to(device)
            noisy_audio = batch["noisy_audio"].to(device)
            clean_audio = batch["clean_audio"].to(device)

            # Algorithm 1, stage 1: only MINE is updated.
            for parameter in mine.parameters():
                parameter.requires_grad = True
            for parameter in model.parameters():
                parameter.requires_grad = False
            if batch_index % grad_accumulation == 0:
                optimizer_m.zero_grad(set_to_none=True)
            loss_m = -mine_bound(mine, language_noise, noisy_audio, clean_audio)
            (loss_m / grad_accumulation).backward()
            if (batch_index + 1) % grad_accumulation == 0 or batch_index + 1 == len(train_loader):
                clip_grad_norm_(mine.parameters(), 1.0)
                optimizer_m.step()

            # Algorithm 1, stage 2: update adapter/tuner, with MINE frozen but
            # differentiable with respect to the tuned language embedding.
            for parameter in mine.parameters():
                parameter.requires_grad = False
            model.freeze_base_parameters()
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if batch_index % grad_accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention,
                    noise_embedding=language_noise,
                    use_cache=False,
                )
                loss_ce = cross_entropy_from_logits(outputs.logits, labels)
                tuned = torch.stack(model._last_noise_states, dim=0).mean(dim=0)
                joint_mi = mine(-tuned, noisy_audio, raw=False).mean()
                loss = loss_ce - float(config["lambda_mi"]) * joint_mi
            scaler.scale(loss / grad_accumulation).backward()
            if (batch_index + 1) % grad_accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model_params, 1.0)
                scaler.step(optimizer)
                scaler.update()

            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"[INFO] epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                    f"mine={joint_mi.item():.4f} mine_stage1={loss_m.item():.4f}"
                )

        eval_loss = evaluate(model, eval_loader, device)
        print(f"[INFO] epoch={epoch + 1} eval_ce={eval_loss:.4f}")
        save_checkpoint(output_dir, "adapter-last.pt", model, mine, config, global_step, eval_loss)
        if eval_loss < best_eval:
            best_eval = eval_loss
            save_checkpoint(output_dir, "adapter-best.pt", model, mine, config, global_step, eval_loss)

    print(f"[INFO] RobustGER training complete: {output_dir}")


if __name__ == "__main__":
    main()
