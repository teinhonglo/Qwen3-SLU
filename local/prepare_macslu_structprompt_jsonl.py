#!/usr/bin/env python3
"""Create PICD-style SLU + PII + CDI data from prepared MAC-SLU JSONL."""

import argparse
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable


PII_PROMPT_TEMPLATE = """你是一个专业的车载系统自然语言理解（NLU）专家。
你的任务是基于用户的查询（Query），完成以下任务：
1.  关系识别 (Pairwise Interaction): 已知当前用户语音中包含的 Domain–Intent 与 Slot 标签，建立每个 Domain–Intent 与其对应 Slot 之间的关系。

你需要严格遵循以下规则：
1.  每个 Slot 必须归属于正确的 Domain–Intent semantic frame。
2.  只输出 Slot 名称，不输出 Slot Value。
3.  按照 semantic frame 首次出现顺序输出。
4.  最终回答中除了指定 JSON，不要包含其他文字。

Domain–Intent 候选：
{domain_intents}

Slot 候选：
{slots}"""

CDI_PROMPT_TEMPLATE = """你是一个专业的车载系统自然语言理解（NLU）专家。
你的任务是基于用户的查询（Query），完成以下任务：
1.  意图数量一致性判断 (Intent Count Verification): 判断当前用户语音与下列参考查询是否包含相同数量的 Intent。

你需要严格遵循以下规则：
1.  只比较 Intent 的数量，不需要输出具体的 Domain、Intent 或 Slot。
2.  如果 Intent 数量相同，请输出 true。
3.  如果 Intent 数量不同，请输出 false。
4.  最终回答中除了 true 或 false，不要包含其他文字。

参考查询：
{reference_query}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PICD-style SLU + PII + CDI MAC-SLU JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--expand-splits",
        nargs="+",
        default=["train"],
        help="Splits that receive PII and CDI auxiliary tasks in addition to full SLU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=66,
        help="Random seed for PII candidate shuffling and CDI reference sampling",
    )
    parser.add_argument(
        "--cdi-pairs-per-class",
        type=int,
        default=1,
        help="Number of positive and negative CDI references sampled per anchor",
    )
    parser.add_argument(
        "--cdi-max-anchor-intent-count",
        type=int,
        default=3,
        help=(
            "Maximum intent count used as a CDI anchor. Rows above this limit "
            "remain available as negative references and in SLU/PII."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def semantic_frames(row: dict) -> list[dict]:
    semantics = row.get("semantics", [])
    if not isinstance(semantics, list):
        raise ValueError("'semantics' must be a JSON List")
    return [frame for frame in semantics if isinstance(frame, dict)]


def unique_domain_intents(frames: Iterable[dict]) -> list[dict]:
    values = []
    seen = set()
    for frame in frames:
        domain = frame.get("domain")
        intent = frame.get("intent")
        if not isinstance(domain, str) or not domain:
            continue
        if not isinstance(intent, str):
            intent = ""
        key = (domain, intent)
        if key in seen:
            continue
        seen.add(key)
        values.append({"domain": domain, "intent": intent})
    return values


def unique_slot_names(frames: Iterable[dict]) -> list[str]:
    names = []
    seen = set()
    for frame in frames:
        for field in ("slots", "implicit_slots"):
            slots = frame.get(field, {})
            if not isinstance(slots, dict):
                continue
            for name in slots:
                name = str(name)
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
    return names


def pairwise_target(frames: Iterable[dict]) -> list[dict]:
    relations: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    for frame in frames:
        domain = frame.get("domain")
        intent = frame.get("intent")
        if not isinstance(domain, str) or not domain:
            continue
        if not isinstance(intent, str):
            intent = ""
        key = (domain, intent)
        slot_names = relations.setdefault(key, [])
        seen = set(slot_names)
        for field in ("slots", "implicit_slots"):
            slots = frame.get(field, {})
            if not isinstance(slots, dict):
                continue
            for name in slots:
                name = str(name)
                if name not in seen:
                    seen.add(name)
                    slot_names.append(name)
    return [
        {"domain": domain, "intent": intent, "slots": slots}
        for (domain, intent), slots in relations.items()
    ]


def intent_count(frames: Iterable[dict]) -> int:
    return len(unique_domain_intents(frames))


def json_target_text(value) -> str:
    return "language None<asr_text>" + json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    )


def bool_target_text(value: bool) -> str:
    return "language None<asr_text>" + ("true" if value else "false")


def slu_row(row: dict) -> dict:
    result = dict(row)
    result["task"] = "slu"
    return result


def pii_row(row: dict, rng: random.Random) -> dict:
    frames = semantic_frames(row)
    domain_intents = unique_domain_intents(frames)
    slots = unique_slot_names(frames)

    domain_intent_candidates = list(domain_intents)
    slot_candidates = list(slots)
    rng.shuffle(domain_intent_candidates)
    rng.shuffle(slot_candidates)

    result = dict(row)
    result["text_id"] = f"{row.get('text_id', '')}__pii"
    result["task"] = "pii"
    result["prompt"] = PII_PROMPT_TEMPLATE.format(
        domain_intents=json.dumps(
            domain_intent_candidates, ensure_ascii=False, separators=(",", ":")
        ),
        slots=json.dumps(slot_candidates, ensure_ascii=False, separators=(",", ":")),
    )
    result["text"] = json_target_text(pairwise_target(frames))
    result["pii_domain_intents"] = domain_intent_candidates
    result["pii_slots"] = slot_candidates
    return result


def build_cdi_rows(
    rows: list[dict],
    rng: random.Random,
    pairs_per_class: int,
    max_anchor_intent_count: int = 3,
) -> list[list[dict]]:
    if len(rows) < 2:
        raise ValueError("CDI requires at least two source rows")
    if pairs_per_class < 1:
        raise ValueError("--cdi-pairs-per-class must be at least 1")
    if max_anchor_intent_count < 0:
        raise ValueError("--cdi-max-anchor-intent-count must be at least 0")

    infos = []
    pools = defaultdict(list)
    for index, row in enumerate(rows):
        frames = semantic_frames(row)
        query = str(row.get("query", "") or "").strip()
        if not query:
            raise ValueError(
                f"CDI requires non-empty query text: {row.get('text_id', index)}"
            )
        count = intent_count(frames)
        info = {
            "index": index,
            "row": row,
            "count": count,
            "query": query,
            "text_id": str(row.get("text_id", "")),
        }
        infos.append(info)
        pools[count].append(info)

    cdi_rows = []
    for current in infos:
        if current["count"] > max_anchor_intent_count:
            cdi_rows.append([])
            continue

        positive = [
            candidate
            for candidate in pools[current["count"]]
            if candidate["index"] != current["index"]
        ]
        negative = [
            candidate
            for count, candidates in pools.items()
            if count != current["count"]
            for candidate in candidates
        ]

        if len(positive) < pairs_per_class:
            raise ValueError(
                "Not enough positive CDI references for "
                f"{current['text_id'] or current['index']}: "
                f"intent_count={current['count']}, available={len(positive)}, "
                f"required={pairs_per_class}"
            )
        if len(negative) < pairs_per_class:
            raise ValueError(
                "Not enough negative CDI references for "
                f"{current['text_id'] or current['index']}: "
                f"intent_count={current['count']}, available={len(negative)}, "
                f"required={pairs_per_class}"
            )

        sampled_pairs = [
            (True, pair_index, reference)
            for pair_index, reference in enumerate(
                rng.sample(positive, pairs_per_class), start=1
            )
        ]
        sampled_pairs.extend(
            (False, pair_index, reference)
            for pair_index, reference in enumerate(
                rng.sample(negative, pairs_per_class), start=1
            )
        )
        rng.shuffle(sampled_pairs)

        current_rows = []
        for label, pair_index, reference in sampled_pairs:
            pair_type = "positive" if label else "negative"
            result = dict(current["row"])
            result["text_id"] = (
                f"{current['text_id']}__cdi_{pair_type}_{pair_index}"
            )
            result["task"] = "cdi"
            result["prompt"] = CDI_PROMPT_TEMPLATE.format(
                reference_query=reference["query"]
            )
            result["text"] = bool_target_text(label)
            result["reference_text_id"] = reference["text_id"]
            result["reference_query"] = reference["query"]
            result["current_intent_count"] = current["count"]
            result["reference_intent_count"] = reference["count"]
            result["cdi_label"] = label
            current_rows.append(result)
        cdi_rows.append(current_rows)

    return cdi_rows


def repeat_rows_to_size(
    rows: list[dict], target_size: int, rng: random.Random, task: str
) -> list[dict]:
    if target_size < len(rows):
        raise ValueError(
            f"Cannot preserve all {task} rows while balancing to CDI: "
            f"source_rows={len(rows)}, cdi_rows={target_size}"
        )

    repeated = []
    while len(repeated) < target_size:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        remaining = target_size - len(repeated)
        repeated.extend(rows[index] for index in indices[:remaining])
    return repeated


def convert_split(
    src_path: Path,
    output_path: Path,
    expand: bool,
    seed: int,
    cdi_pairs_per_class: int,
    cdi_max_anchor_intent_count: int = 3,
) -> dict[str, int]:
    source_rows = load_jsonl(src_path)
    counts = {
        "source_rows": len(source_rows),
        "slu": 0,
        "pii": 0,
        "cdi": 0,
        "cdi_true": 0,
        "cdi_false": 0,
        "cdi_anchors": 0,
        "total": 0,
    }

    if expand:
        pii_rng = random.Random(seed)
        cdi_rng = random.Random(seed + 1)
        pii_rows = [pii_row(row, pii_rng) for row in source_rows]
        cdi_rows = build_cdi_rows(
            source_rows,
            cdi_rng,
            pairs_per_class=cdi_pairs_per_class,
            max_anchor_intent_count=cdi_max_anchor_intent_count,
        )
        counts["cdi_anchors"] = sum(bool(group) for group in cdi_rows)
        flat_cdi_rows = [row for group in cdi_rows for row in group]
        slu_rows = repeat_rows_to_size(
            [slu_row(row) for row in source_rows],
            len(flat_cdi_rows),
            random.Random(seed + 2),
            "SLU",
        )
        balanced_pii_rows = repeat_rows_to_size(
            pii_rows,
            len(flat_cdi_rows),
            random.Random(seed + 3),
            "PII",
        )
    else:
        pii_rows = []
        cdi_rows = []
        flat_cdi_rows = []
        slu_rows = []
        balanced_pii_rows = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        if expand:
            output_groups = zip(slu_rows, balanced_pii_rows, flat_cdi_rows)
        else:
            output_groups = ((slu_row(row),) for row in source_rows)

        for rows in output_groups:
            for result in rows:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["task"]] += 1
                if result["task"] == "cdi":
                    counts["cdi_true" if result["cdi_label"] else "cdi_false"] += 1
                counts["total"] += 1

    expected = 3 * counts["cdi"] if expand else counts["source_rows"]
    if counts["total"] != expected:
        raise RuntimeError(
            f"Sanity check failed for {src_path}: total={counts['total']}, expected={expected}"
        )
    return counts


def main() -> None:
    args = parse_args()

    src_root = Path(args.src_json_root)
    output_root = Path(args.json_root)
    expand_splits = set(args.expand_splits)
    unknown_splits = expand_splits.difference(args.splits)
    if unknown_splits:
        raise ValueError(f"--expand-splits not present in --splits: {sorted(unknown_splits)}")

    for split_index, split in enumerate(args.splits):
        src_path = src_root / f"{split}.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"Required source JSONL not found: {src_path}")
        counts = convert_split(
            src_path,
            output_root / f"{split}.jsonl",
            split in expand_splits,
            args.seed + split_index * 10000,
            args.cdi_pairs_per_class,
            args.cdi_max_anchor_intent_count,
        )
        print(f"[INFO] {split}:")
        print(f"source_rows={counts['source_rows']}")
        print(f"slu={counts['slu']}")
        print(f"pii={counts['pii']}")
        print(f"cdi={counts['cdi']}")
        if counts["cdi"]:
            print(f"cdi_pairs_per_class={args.cdi_pairs_per_class}")
            print(f"cdi_max_anchor_intent_count={args.cdi_max_anchor_intent_count}")
            print(f"cdi_anchors={counts['cdi_anchors']}")
            print(f"cdi_true={counts['cdi_true']}")
            print(f"cdi_false={counts['cdi_false']}")
        print(f"total={counts['total']}")


if __name__ == "__main__":
    main()
