"""Shared domain/intent prototype prompt formatting utilities."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from slu_decoding.prototypes import MACSLULabelSchema, parse_semantics_field


def unique_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def format_domain_intent_candidates(
    base_prompt: str,
    domains: Sequence[str],
    intents: Sequence[str],
    domain_title: str = "Candidate domains",
    intent_title: str = "Candidate intents",
    separator: str = ", ",
) -> str:
    """Append domain/intent candidates to a prompt using one train/test format."""
    domain_text = separator.join(str(x) for x in domains)
    intent_text = separator.join(str(x) for x in intents)
    return (
        f"{(base_prompt or '').rstrip()}\n\n"
        f"{domain_title}: {domain_text}\n"
        f"{intent_title}: {intent_text}"
    )


def extract_gold_domain_intents(example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Extract gold domain and intent labels from a MAC-SLU style example."""
    semantics = example.get("semantics", [])
    if not semantics and "text" in example:
        try:
            obj = json.loads(example.get("text") or "{}")
            semantics = obj.get("semantics", [])
        except Exception:
            semantics = []
    frames = parse_semantics_field(semantics)
    domains = unique_keep_order(frame.get("domain", "") for frame in frames)
    intents = unique_keep_order(frame.get("intent", "") for frame in frames)
    return domains, intents


def flatten_schema_intents(schema: MACSLULabelSchema) -> List[str]:
    intents: List[str] = []
    for domain in schema.valid_domains():
        intents.extend(schema.valid_intents(domain))
    return unique_keep_order(intents)


def pad_candidates(
    gold: Sequence[str],
    pool: Sequence[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    """Keep gold labels, randomly pad to k from pool, and shuffle order."""
    k = int(k)
    candidates = unique_keep_order(gold)
    pool_values = [x for x in unique_keep_order(pool) if x not in candidates]
    rng.shuffle(pool_values)
    for value in pool_values:
        if len(candidates) >= k:
            break
        candidates.append(value)
    if k > 0:
        candidates = candidates[:k]
    rng.shuffle(candidates)
    return candidates


def build_training_candidate_labels(
    example: Dict[str, Any],
    schema: MACSLULabelSchema,
    k: int,
    rng: random.Random,
    domain_aware_intents: bool = True,
) -> Tuple[List[str], List[str], str, str]:
    """Build train-time candidates that include gold labels and random labels up to k."""
    gold_domains, gold_intents = extract_gold_domain_intents(example)
    all_domains = schema.valid_domains()
    if domain_aware_intents and gold_domains:
        intent_pool: List[str] = []
        for domain in gold_domains:
            intent_pool.extend(schema.valid_intents(domain))
        intent_pool.extend(flatten_schema_intents(schema))
        intent_pool = unique_keep_order(intent_pool)
    else:
        intent_pool = flatten_schema_intents(schema)

    domains = pad_candidates(gold_domains, all_domains, k, rng)
    intents = pad_candidates(gold_intents, intent_pool, k, rng)
    gold_domain = gold_domains[0] if gold_domains else ""
    gold_intent = gold_intents[0] if gold_intents else ""
    return domains, intents, gold_domain, gold_intent


def get_prompt_template(prototype_conf: Dict[str, Any]) -> Dict[str, str]:
    template = dict((prototype_conf or {}).get("prompt_template", {}) or {})
    return {
        "domain_title": str(template.get("domain_title", "Candidate domains")),
        "intent_title": str(template.get("intent_title", "Candidate intents")),
        "separator": str(template.get("separator", ", ")),
    }
