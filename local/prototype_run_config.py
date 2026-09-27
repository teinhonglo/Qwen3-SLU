#!/usr/bin/env python3
"""Resolve prototype defaults and materialize runtime training configs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VALID_SOURCES = {"audio_only", "audio_prompt", "audio_prefix", "text_prefix"}
VALID_POOLING = {"mean_pooling", "last_hidden_state"}
VALID_LOSS_TYPES = {"bce"}


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
    required = ("k", "prototype_source", "pooling")
    missing = [key for key in required if key not in prototype]
    if missing:
        raise KeyError(
            "prototype_train_conf is missing model_args.prototype defaults: "
            + ", ".join(missing)
        )

    top_k = int(prototype["k"])
    source = str(prototype["prototype_source"])
    pooling = str(prototype["pooling"])
    loss_type = str(prototype.get("loss_type", "bce")).lower()
    if top_k <= 0:
        raise ValueError("prototype k must be a positive integer")
    if source not in VALID_SOURCES:
        raise ValueError(f"unsupported prototype_source: {source}")
    if pooling not in VALID_POOLING:
        raise ValueError(f"unsupported prototype pooling: {pooling}")
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(f"unsupported prototype loss_type: {loss_type}")
    if not bool(prototype.get("normalize", True)):
        raise ValueError("prototype BCE requires prototype.normalize=true")
    scale_init = float(prototype.get("logit_scale_init", 10.0))
    scale_max = float(prototype.get("logit_scale_max", 100.0))
    if scale_init <= 0.0 or scale_max < scale_init:
        raise ValueError(
            "scaled-cosine BCE requires 0 < logit_scale_init <= logit_scale_max"
        )
    return {
        "top_k": top_k,
        "source": source,
        "pooling": pooling,
    }


def command_defaults(args: argparse.Namespace) -> None:
    defaults = resolve_defaults(args.config)
    print(defaults["top_k"])
    print(defaults["source"])
    print(defaults["pooling"])


def command_materialize(args: argparse.Namespace) -> None:
    config = load_train_conf(args.config)
    training_args, model_args = config
    defaults = resolve_defaults(args.config)
    best_metric = str(training_args.get("metric_for_best_model", ""))
    if not best_metric or best_metric.startswith("eval_domain_intent_all_gold_covered@"):
        training_args["metric_for_best_model"] = (
            f"eval_domain_intent_all_gold_covered@{defaults['top_k']}"
        )

    prototype = dict(model_args.get("prototype", {}) or {})
    prototype.update(
        {
            "enabled": True,
            "labels_path": args.labels_path,
            "schema_path": args.schema_path,
            "prototype_json": args.prototype_json,
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
    materialize.set_defaults(func=command_materialize)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.func(parsed_args)
