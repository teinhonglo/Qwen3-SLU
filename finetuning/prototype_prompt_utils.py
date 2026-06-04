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
    """Keep *all* gold labels, then randomly pad to at least k candidates.

    Multi-domain / multi-intent examples may contain more than k gold labels.
    In that case we keep every gold label instead of truncating positives out of
    the prompt.
    """
    k = int(k)
    candidates = unique_keep_order(gold)
    target_len = max(k, len(candidates)) if k > 0 else len(candidates)
    pool_values = [x for x in unique_keep_order(pool) if x not in candidates]
    rng.shuffle(pool_values)
    for value in pool_values:
        if len(candidates) >= target_len:
            break
        candidates.append(value)
    rng.shuffle(candidates)
    return candidates


def build_training_candidate_labels(
    example: Dict[str, Any],
    schema: MACSLULabelSchema,
    k: int,
    rng: random.Random,
    domain_aware_intents: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Build train-time candidates and return all gold domain/intent labels."""
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
    return domains, intents, gold_domains, gold_intents


def get_prompt_template(prototype_conf: Dict[str, Any]) -> Dict[str, str]:
    template = dict((prototype_conf or {}).get("prompt_template", {}) or {})
    return {
        "domain_title": str(template.get("domain_title", "Candidate domains")),
        "intent_title": str(template.get("intent_title", "Candidate intents")),
        "separator": str(template.get("separator", ", ")),
    }
