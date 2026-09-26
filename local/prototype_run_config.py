#!/usr/bin/env python3
"""Resolve prototype defaults and materialize runtime training configs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VALID_SOURCES = {"audio_only", "audio_prompt", "audio_prefix", "text_prefix"}
VALID_POOLING = {"mean_pooling", "last_hidden_state"}


def load_train_conf(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, list) or len(config) != 2:
        raise ValueError("prototype_train_conf must be [training_args, model_args]")
    if not all(isinstance(section, dict) for section in config):
        raise ValueError("prototype_train_conf entries must be dictionaries")
    return config


def resolve_defaults(path: str) -> Dict[str, Any]:
    _, model_args = load_train_conf(path)
    prototype = model_args.get("prototype", {}) or {}
    required = ("k", "metric_ks", "prototype_source", "pooling")
    missing = [key for key in required if key not in prototype]
    if missing:
        raise KeyError(
            "prototype_train_conf is missing model_args.prototype defaults: "
            + ", ".join(missing)
        )

    top_k = int(prototype["k"])
    metric_ks = [int(value) for value in prototype["metric_ks"]]
    source = str(prototype["prototype_source"])
    pooling = str(prototype["pooling"])
    if top_k <= 0 or not metric_ks or any(value <= 0 for value in metric_ks):
        raise ValueError("prototype k and metric_ks must contain positive integers")
    if source not in VALID_SOURCES:
        raise ValueError(f"unsupported prototype_source: {source}")
    if pooling not in VALID_POOLING:
        raise ValueError(f"unsupported prototype pooling: {pooling}")

    lora_type = str(model_args.get("lora_type", "default")).lower()
    lora_config = model_args.get("lora_config")
    if lora_type == "adapter_head":
        finetune_type = "adapter_head"
    elif lora_type == "qlora":
        finetune_type = "qlora"
    elif lora_config:
        finetune_type = "lora"
    else:
        finetune_type = "full_ft"
    return {
        "finetune_type": finetune_type,
        "top_k": top_k,
        "metric_ks": metric_ks,
        "source": source,
        "pooling": pooling,
    }


def command_defaults(args: argparse.Namespace) -> None:
    defaults = resolve_defaults(args.config)
    print(defaults["finetune_type"])
    print(defaults["top_k"])
    print(" ".join(str(value) for value in defaults["metric_ks"]))
    print(defaults["source"])
    print(defaults["pooling"])


def command_materialize(args: argparse.Namespace) -> None:
    config = load_train_conf(args.config)
    training_args, model_args = config
    best_metric = str(training_args.get("metric_for_best_model", ""))
    if not best_metric or best_metric.startswith("eval_domain_intent_all_gold_covered@"):
        training_args["metric_for_best_model"] = (
            f"eval_domain_intent_all_gold_covered@{args.top_k}"
        )

    prototype = dict(model_args.get("prototype", {}) or {})
    prototype.update(
        {
            "enabled": True,
            "labels_path": args.labels_path,
            "schema_path": args.schema_path,
            "prototype_json": args.prototype_json,
            "k": args.top_k,
            "metric_ks": args.metric_ks,
            "prototype_source": args.source,
            "pooling": args.pooling,
        }
    )
    prototype.pop("init_path", None)
    model_args["prototype"] = prototype

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        f.write("\n")
    print(f"[info] wrote prototype runtime config: {output}; init_json={args.prototype_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    defaults = subparsers.add_parser("defaults")
    defaults.add_argument("--config", required=True)
    defaults.set_defaults(func=command_defaults)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--labels-path", required=True)
    materialize.add_argument("--schema-path", required=True)
    materialize.add_argument("--prototype-json", required=True)
    materialize.add_argument("--top-k", type=int, required=True)
    materialize.add_argument("--metric-ks", nargs="+", type=int, required=True)
    materialize.add_argument("--source", choices=sorted(VALID_SOURCES), required=True)
    materialize.add_argument("--pooling", choices=sorted(VALID_POOLING), required=True)
    materialize.set_defaults(func=command_materialize)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.func(parsed_args)
