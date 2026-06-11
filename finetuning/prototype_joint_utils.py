"""Shared helpers for MAC-SLU joint domain-intent prototype labels."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from slu_decoding.prototypes import parse_semantics_field

DOMAIN_INTENT_SEP = "|||"


def make_domain_intent_label(domain: str, intent: str) -> str:
    return f"{str(domain).strip()}{DOMAIN_INTENT_SEP}{str(intent).strip()}"


def split_domain_intent_label(label: str) -> Tuple[str, str]:
    parts = str(label).split(DOMAIN_INTENT_SEP, 1)
    if len(parts) != 2:
        return str(label), ""
    return parts[0], parts[1]


def unique_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def extract_semantic_frames(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract parsed semantic frames from a MAC-SLU style example."""
    semantics = example.get("semantics", [])
    if not semantics and "text" in example:
        try:
            obj = json.loads(example.get("text") or "{}")
            semantics = obj.get("semantics", [])
        except Exception:
            semantics = []
    return parse_semantics_field(semantics)


def extract_gold_domain_intent_labels(example: Dict[str, Any]) -> List[str]:
    """Extract gold joint domain-intent labels from a MAC-SLU style example."""
    frames = extract_semantic_frames(example)
    return unique_keep_order(
        make_domain_intent_label(frame.get("domain", ""), frame.get("intent", ""))
        for frame in frames
        if frame.get("domain", "") and frame.get("intent", "")
    )
