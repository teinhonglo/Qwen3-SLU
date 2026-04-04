#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read semantics from preprocessed JSONL and output\n"
            "1) domain -> intents\n"
            "2) domain-intent -> slots\n"
            "as JSONL mapping files."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to preprocessed JSONL file (one JSON object per line)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory. Two files will be created inside it: "
            "'domain_intents.jsonl' and 'domain_intent_slots.jsonl'."
        ),
    )
    return parser.parse_args()


def _normalize_semantics(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []

    return []


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"JSON parse failed: {path}:{line_no}: {err}") from err

            if not isinstance(row, dict):
                continue
            yield row


def extract_mappings(
    rows: Iterable[dict],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    domain_to_intents: Dict[str, Set[str]] = defaultdict(set)
    domain_intent_to_slots: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for row in rows:
        semantics = _normalize_semantics(row.get("semantics", []))

        for item in semantics:
            if not isinstance(item, dict):
                continue

            domain = item.get("domain")
            intent = item.get("intent")
            slots = item.get("slots", {})

            if not isinstance(domain, str) or not domain.strip():
                continue
            domain = domain.strip()

            if isinstance(intent, str) and intent.strip():
                intent = intent.strip()
                domain_to_intents[domain].add(intent)
            else:
                intent = ""

            # Ensure each (domain, intent) appears exactly once in output,
            # even when it has no slots.
            _ = domain_intent_to_slots[(domain, intent)]

            if not isinstance(slots, dict):
                continue

            for slot_value in slots.values():
                if isinstance(slot_value, str):
                    slot_value = slot_value.strip()
                    if slot_value:
                        domain_intent_to_slots[(domain, intent)].add(slot_value)
                elif slot_value is not None:
                    domain_intent_to_slots[(domain, intent)].add(str(slot_value))

    return domain_to_intents, domain_intent_to_slots


def write_domain_intents_jsonl(path: Path, mapping: Dict[str, Set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for domain in sorted(mapping.keys()):
            record = {
                "domain": domain,
                "intents": sorted(mapping[domain]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_domain_intent_slots_jsonl(
    path: Path,
    mapping: Dict[Tuple[str, str], Set[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for (domain, intent) in sorted(mapping.keys()):
            record = {
                "domain_intent": f"{domain}-{intent}" if intent else f"{domain}-",
                "slots": sorted(mapping[(domain, intent)]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    domain_intents_out = output_dir / "domain_intents.jsonl"
    domain_intent_slots_out = output_dir / "domain_intent_slots.jsonl"

    rows = list(read_jsonl(input_jsonl))
    domain_to_intents, domain_intent_to_slots = extract_mappings(rows)

    write_domain_intents_jsonl(domain_intents_out, domain_to_intents)
    write_domain_intent_slots_jsonl(domain_intent_slots_out, domain_intent_to_slots)

    print(f"[INFO] input rows: {len(rows)}")
    print(f"[INFO] domain->intents: {len(domain_to_intents)} domains")
    print(f"[INFO] domain-intent->slots: {len(domain_intent_to_slots)} pairs")
    print(f"[INFO] wrote: {domain_intents_out}")
    print(f"[INFO] wrote: {domain_intent_slots_out}")


if __name__ == "__main__":
    main()
