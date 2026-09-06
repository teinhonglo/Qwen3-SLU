#!/usr/bin/env python3
"""Build MAC-SLU schema for optional DExperts decoding."""

import argparse
import json
import os
from collections import defaultdict

SEP = "|||"
TARGET_MARKER = "<asr_text>"


def _encoded_inner_string(value):
    """Encode one inner-JSON string exactly as it appears in outer JSON text."""
    inner_literal = json.dumps(str(value), ensure_ascii=False)
    return json.dumps(inner_literal, ensure_ascii=False)[1:-1]


def _find_literal(text, value, start):
    literal = _encoded_inner_string(value)
    offset = text.find(literal, start)
    if offset < 0:
        raise ValueError(f"missing serialized literal after offset {start}: {value!r}")
    # Encoded inner string literals start/end with the two characters \".
    return {
        "start": offset,
        "content_start": offset + 2,
        "content_end": offset + len(literal) - 2,
        "end": offset + len(literal),
    }


def _add_surface(forms, name, value):
    if value:
        forms[name].add(value)


def _find_object(text, name, start):
    key = _find_literal(text, name, start)
    colon = text.find(":", key["end"])
    opener = text.find("{", colon + 1)
    if colon < 0 or opener < 0:
        raise ValueError(f"missing object opener for {name!r}")
    return key, opener + 1


def _extract_object_surfaces(text, items, content_start, forms, prefix):
    """Extract observed key/value separators and next-key/object-close forms."""
    cursor = content_start
    previous_value_end = None
    pairs = []
    for key_name, value in items:
        key = _find_literal(text, key_name, cursor)
        value_span = _find_literal(text, value, key["end"])
        pairs.append((key, value_span))
        _add_surface(
            forms,
            f"{prefix}_key_followups",
            text[key["content_end"] : value_span["content_start"]],
        )
        _add_surface(
            forms,
            f"after_{prefix}_key",
            text[key["end"] : value_span["content_start"]],
        )
        if previous_value_end is not None:
            _add_surface(
                forms,
                f"{prefix}_next",
                text[previous_value_end : key["content_start"]],
            )
        previous_value_end = value_span["end"]
        cursor = value_span["end"]

    close = text.find("}", cursor)
    if close < 0:
        raise ValueError(f"missing closing brace for {prefix!r}")
    first_boundary = pairs[0][0]["content_start"] if pairs else close + 1
    _add_surface(forms, f"{prefix}_initial", text[content_start:first_boundary])
    if previous_value_end is not None:
        _add_surface(forms, f"{prefix}_next", text[previous_value_end : close + 1])
    return close + 1


def extract_surface_forms(target_text, frames):
    """Extract constraint literals from one real serialized training target.

    Semantic labels and slot values are treated as variable spans. Everything
    between those spans is copied verbatim from ``row['text']``.
    """
    if TARGET_MARKER not in target_text:
        raise ValueError(f"target does not contain {TARGET_MARKER!r}")
    payload = target_text.split(TARGET_MARKER, 1)[1]
    outer = json.loads(payload)
    semantics_text = outer.get("semantics")
    if not isinstance(semantics_text, str):
        raise ValueError("outer payload semantics must be a JSON string")
    parsed_frames = json.loads(semantics_text)
    if parsed_frames != frames:
        raise ValueError("row semantics and text target semantics disagree")

    semantics_literal = json.dumps(semantics_text, ensure_ascii=False)
    semantics_key = payload.find(json.dumps("semantics", ensure_ascii=False))
    literal_start = payload.find(semantics_literal, max(0, semantics_key))
    if literal_start < 0:
        raise ValueError("cannot locate serialized semantics value in target")
    content_start = literal_start + 1
    content_end = literal_start + len(semantics_literal) - 1

    forms = defaultdict(set)
    if not frames:
        _add_surface(forms, "semantics_start", payload[content_start:])
        return forms

    cursor = content_start
    for frame_index, frame in enumerate(frames):
        domain_key = _find_literal(payload, "domain", cursor)
        domain = _find_literal(payload, frame["domain"], domain_key["end"])
        if frame_index == 0:
            _add_surface(
                forms,
                "semantics_start",
                payload[content_start : domain["content_start"]],
            )
        else:
            _add_surface(
                forms,
                "after_implicit_slots",
                payload[cursor : domain["content_start"]],
            )

        intent_key = _find_literal(payload, "intent", domain["end"])
        intent = _find_literal(payload, frame["intent"], intent_key["end"])
        _add_surface(
            forms,
            "domain_followups",
            payload[domain["content_end"] : intent["content_start"]],
        )
        _add_surface(
            forms,
            "after_domain",
            payload[domain["end"] : intent["content_start"]],
        )

        _, slots_start = _find_object(payload, "slots", intent["end"])
        _add_surface(
            forms,
            "intent_followups",
            payload[intent["content_end"] : slots_start],
        )
        _add_surface(
            forms,
            "after_intent",
            payload[intent["end"] : slots_start],
        )
        slots_end = _extract_object_surfaces(
            payload,
            list((frame.get("slots") or {}).items()),
            slots_start,
            forms,
            "slots",
        )

        _, implicit_start = _find_object(payload, "implicit_slots", slots_end)
        _add_surface(
            forms,
            "after_slots",
            payload[slots_end:implicit_start],
        )
        implicit_end = _extract_object_surfaces(
            payload,
            list((frame.get("implicit_slots") or {}).items()),
            implicit_start,
            forms,
            "implicit_slots",
        )
        cursor = implicit_end

    _add_surface(forms, "after_implicit_slots", payload[cursor:])
    if content_end >= len(payload):
        raise ValueError("invalid serialized semantics boundary")
    return forms


def iter_rows(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as exc:
                    print(f"[warn] bad json {path}:{line_id}: {exc}")


def build_schema(paths):
    domains = set()
    domain2intents = defaultdict(set)
    di2slot = defaultdict(set)
    di2implicit = defaultdict(set)
    surface_forms = defaultdict(set)
    surface_errors = []

    for row in iter_rows(paths):
        frames = row.get("semantics", []) or []
        if isinstance(frames, str):
            try:
                frames = json.loads(frames)
            except Exception:
                frames = []

        target_text = row.get("text", "")
        if not target_text:
            surface_errors.append(
                {"text_id": str(row.get("text_id", "")), "error": "missing row['text']"}
            )
        else:
            try:
                observed = extract_surface_forms(target_text, frames)
                for name, values in observed.items():
                    surface_forms[name].update(values)
            except Exception as exc:
                surface_errors.append(
                    {"text_id": str(row.get("text_id", "")), "error": str(exc)}
                )

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            domain = frame.get("domain", "")
            intent = frame.get("intent", "")
            if not domain or not intent:
                continue

            domains.add(domain)
            domain2intents[domain].add(intent)
            di_key = f"{domain}{SEP}{intent}"

            for slot_key in (frame.get("slots", {}) or {}).keys():
                di2slot[di_key].add(slot_key)
            for implicit_key in (frame.get("implicit_slots", {}) or {}).keys():
                di2implicit[di_key].add(implicit_key)

    if surface_errors:
        first = surface_errors[0]
        raise ValueError(
            f"failed to extract target format for {len(surface_errors)} row(s); "
            f"first text_id={first['text_id']!r}: {first['error']}"
        )
    if not surface_forms:
        raise ValueError("no real row['text'] target formats were observed")

    return {
        "domains": sorted(domains),
        "domain2intents": {k: sorted(v) for k, v in sorted(domain2intents.items())},
        "domain_intent2slot_keys": {k: sorted(v) for k, v in sorted(di2slot.items())},
        "domain_intent2implicit_slot_keys": {
            k: sorted(v) for k, v in sorted(di2implicit.items())
        },
        "constraint_surface_forms": {
            k: sorted(v) for k, v in sorted(surface_forms.items())
        },
        "constraint_sources": list(paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonls", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    for path in args.input_jsonls:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    schema = build_schema(args.input_jsonls)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"[info] saved schema: {args.output_json}")


if __name__ == "__main__":
    main()
