#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

DOMAIN_ACTION_PATTERNS = {
    "车载控制": [
        "打开", "关闭", "调到", "调成", "调低", "调高", "调大", "调小",
        "设为", "设置", "开一下", "关一下",
        "空调", "天窗", "车窗", "座椅", "HUD", "氛围灯", "大灯", "阅读灯"
    ],
    "地图": [
        "导航", "带我去", "去", "路线", "路况", "目的地", "途经点",
        "附近", "周边", "加油站", "停车场", "充电站"
    ],
    "音乐": ["播放", "放一首", "来一首", "听", "歌曲", "音乐", "歌手"],
    "收音机": ["收音机", "电台", "FM", "AM", "调频", "频道"],
    "打电话": ["打电话", "拨打", "呼叫", "回拨", "接电话", "挂电话"],
    "天气": [
        "天气", "温度", "下雨", "下雪", "空气质量", "湿度", "风力",
        "紫外线", "穿什么"
    ],
    "影视": ["播放", "看", "电影", "电视剧", "视频", "综艺", "第几集", "快进"],
}

TEMP_HINT_KEYWORDS = ["空调", "温度", "制冷", "制热"]
NUMERIC_VALUE_PATTERN = re.compile(r"^[0-9]+$|^[零一二三四五六七八九十百千万两〇]+$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--write_reports", action="store_true")
    return p.parse_args()


def load_labels_schema(labels_path: Path):
    lines = labels_path.read_text(encoding="utf-8").splitlines()
    valid_domains = set()
    valid_intents_by_domain = defaultdict(set)
    valid_slots_by_domain = defaultdict(set)

    section = None
    current_domain = None

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("DOMAIN_INTENT_LIST"):
            section = "domain_intent"
            continue
        if stripped.startswith("SLOT_LIST"):
            section = "slot_list"
            continue

        if section == "domain_intent":
            if line.startswith("- "):
                current_domain = stripped[2:].strip()
                valid_domains.add(current_domain)
                valid_intents_by_domain[current_domain]
            elif line.startswith("    - ") and current_domain:
                intent = stripped[2:].strip()
                valid_intents_by_domain[current_domain].add(intent)

        elif section == "slot_list":
            if not stripped.startswith("-"):
                continue
            item = stripped[1:].strip()
            left = item.split(":", 1)[0].strip()
            if "-" not in left:
                continue
            domain, slot = left.split("-", 1)
            domain = domain.strip()
            slot = slot.strip()
            if domain:
                valid_domains.add(domain)
                valid_slots_by_domain[domain].add(slot)

    return valid_domains, dict(valid_intents_by_domain), dict(valid_slots_by_domain)


def matched_domains(query: str):
    res = set()
    for domain, patterns in DOMAIN_ACTION_PATTERNS.items():
        if any(p in query for p in patterns):
            res.add(domain)
    return res


def append_manual(report_rows, split, text_id, query, issue_type, semantics, reason):
    report_rows.append({
        "split": split,
        "text_id": text_id,
        "query": query,
        "issue_type": issue_type,
        "original_semantics": json.dumps(semantics, ensure_ascii=False),
        "reason": reason,
    })


def maybe_fix_temp(frame, query):
    fixes = 0
    for key in ("slots", "implicit_slots"):
        container = frame.get(key)
        if not isinstance(container, dict):
            continue
        if frame.get("domain") != "车载控制":
            continue
        if "value" not in container:
            continue
        value = str(container.get("value", ""))
        if "度" in value:
            continue
        if not NUMERIC_VALUE_PATTERN.match(value):
            continue
        if f"{value}度" not in query:
            continue

        slots = frame.get("slots") if isinstance(frame.get("slots"), dict) else {}
        implicit_slots = frame.get("implicit_slots") if isinstance(frame.get("implicit_slots"), dict) else {}
        temp_related = (
            slots.get("调节内容") == "温度"
            or implicit_slots.get("调节内容") == "温度"
            or any(k in query for k in TEMP_HINT_KEYWORDS)
        )
        if not temp_related:
            continue

        container["value"] = f"{value}度"
        fixes += 1
    return fixes


def process_split(split, input_dir: Path, output_dir: Path, schema, auto_rows, manual_rows, issue_counts):
    valid_domains, valid_intents_by_domain, valid_slots_by_domain = schema
    in_path = input_dir / f"{split}.jsonl"
    out_path = output_dir / f"{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    auto_fixed_rows = 0
    manual_review_rows = 0
    deduped_frames = 0
    temperature_unit_fixes = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            text_id = row.get("text_id", "")
            query = row.get("query", "")
            semantics = row.get("semantics", [])
            original_semantics = deepcopy(semantics)

            changed = False
            row_manual_issues = 0

            if semantics == []:
                matched = matched_domains(query)
                if matched:
                    row_manual_issues += 1
                    issue_counts["empty_semantics_possible_leak"] += 1
                    append_manual(
                        manual_rows, split, text_id, query,
                        "empty_semantics_possible_leak", original_semantics,
                        f"semantics is empty but query matches domains: {','.join(sorted(matched))}"
                    )

            if isinstance(semantics, list):
                seen = set()
                deduped = []
                for frame in semantics:
                    canonical = json.dumps(frame, ensure_ascii=False, sort_keys=True)
                    if canonical in seen:
                        changed = True
                        deduped_frames += 1
                    else:
                        seen.add(canonical)
                        deduped.append(frame)
                semantics = deduped

                new_semantics = []
                for idx, frame in enumerate(semantics):
                    if not isinstance(frame, dict):
                        row_manual_issues += 1
                        issue_counts["schema_invalid_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_domain", original_semantics, f"frame[{idx}] is not dict")
                        new_semantics.append(frame)
                        continue

                    frame = deepcopy(frame)
                    if "slots" not in frame:
                        frame["slots"] = {}
                        changed = True
                    if "implicit_slots" not in frame:
                        frame["implicit_slots"] = {}
                        changed = True

                    if "domain" not in frame:
                        row_manual_issues += 1
                        issue_counts["schema_invalid_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_domain", original_semantics, f"frame[{idx}] missing domain")
                    if "intent" not in frame:
                        row_manual_issues += 1
                        issue_counts["schema_invalid_intent_for_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_intent_for_domain", original_semantics, f"frame[{idx}] missing intent")

                    if not isinstance(frame.get("slots"), dict):
                        row_manual_issues += 1
                        issue_counts["schema_invalid_slot_for_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_slot_for_domain", original_semantics, f"frame[{idx}] slots is not dict")
                    if not isinstance(frame.get("implicit_slots"), dict):
                        row_manual_issues += 1
                        issue_counts["schema_invalid_slot_for_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_slot_for_domain", original_semantics, f"frame[{idx}] implicit_slots is not dict")

                    domain = frame.get("domain")
                    intent = frame.get("intent")

                    if domain is not None and domain not in valid_domains:
                        row_manual_issues += 1
                        issue_counts["schema_invalid_domain"] += 1
                        append_manual(manual_rows, split, text_id, query, "schema_invalid_domain", original_semantics, f"frame[{idx}] domain={domain} not in labels schema")
                    if domain in valid_domains and intent is not None:
                        if intent not in valid_intents_by_domain.get(domain, set()):
                            row_manual_issues += 1
                            issue_counts["schema_invalid_intent_for_domain"] += 1
                            append_manual(manual_rows, split, text_id, query, "schema_invalid_intent_for_domain", original_semantics, f"frame[{idx}] intent={intent} invalid for domain={domain}")

                    for slot_container_name in ("slots", "implicit_slots"):
                        slot_container = frame.get(slot_container_name)
                        if not isinstance(slot_container, dict):
                            continue
                        for slot_name, slot_value in slot_container.items():
                            if domain in valid_domains and slot_name not in valid_slots_by_domain.get(domain, set()):
                                row_manual_issues += 1
                                issue_counts["schema_invalid_slot_for_domain"] += 1
                                append_manual(manual_rows, split, text_id, query, "schema_invalid_slot_for_domain", original_semantics, f"frame[{idx}] {slot_container_name}.{slot_name} invalid for domain={domain}")
                            if domain == "车载控制" and slot_name == "车内灯类型" and "轮廓灯" in str(slot_value):
                                row_manual_issues += 1
                                issue_counts["suspicious_legal_slot_semantics"] += 1
                                append_manual(manual_rows, split, text_id, query, "suspicious_legal_slot_semantics", original_semantics, f"frame[{idx}] 车内灯类型 contains 轮廓灯")

                    fixes = maybe_fix_temp(frame, query)
                    if fixes:
                        changed = True
                        temperature_unit_fixes += fixes

                    new_semantics.append(frame)

                semantics = new_semantics

                sem_domains = {f.get("domain") for f in semantics if isinstance(f, dict) and isinstance(f.get("domain"), str)}
                matched = matched_domains(query)
                if len(matched) >= 2 and not matched.issubset(sem_domains):
                    row_manual_issues += 1
                    issue_counts["missing_frame_multi_intent_candidate"] += 1
                    append_manual(
                        manual_rows, split, text_id, query,
                        "missing_frame_multi_intent_candidate", original_semantics,
                        f"query matched domains={','.join(sorted(matched))}, semantics domains={','.join(sorted(sem_domains))}"
                    )

            if changed:
                auto_fixed_rows += 1
                auto_rows.append({
                    "split": split,
                    "text_id": text_id,
                    "query": query,
                    "rule_id": "dedupe_exact_semantic_frame;fix_vehicle_temperature_unit",
                    "before_semantics": json.dumps(original_semantics, ensure_ascii=False),
                    "after_semantics": json.dumps(semantics, ensure_ascii=False),
                })

            if row_manual_issues:
                manual_review_rows += 1

            row["semantics"] = semantics
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_rows": total_rows,
        "auto_fixed_rows": auto_fixed_rows,
        "manual_review_rows": manual_review_rows,
        "deduped_frames": deduped_frames,
        "temperature_unit_fixes": temperature_unit_fixes,
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, output_dir / "labels.txt")

    schema = load_labels_schema(labels_path)
    auto_rows = []
    manual_rows = []
    issue_counts = Counter({
        "schema_invalid_domain": 0,
        "schema_invalid_intent_for_domain": 0,
        "schema_invalid_slot_for_domain": 0,
        "empty_semantics_possible_leak": 0,
        "missing_frame_multi_intent_candidate": 0,
        "suspicious_legal_slot_semantics": 0,
    })

    split_stats = {}
    for split in args.splits:
        split_stats[split] = process_split(split, input_dir, output_dir, schema, auto_rows, manual_rows, issue_counts)

    if args.write_reports:
        reports_dir = output_dir / "reports"
        write_csv(
            reports_dir / "auto_fix_report.csv",
            auto_rows,
            ["split", "text_id", "query", "rule_id", "before_semantics", "after_semantics"],
        )
        write_csv(
            reports_dir / "manual_review.csv",
            manual_rows,
            ["split", "text_id", "query", "issue_type", "original_semantics", "reason"],
        )
        summary = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "splits": split_stats,
            "issue_counts": dict(issue_counts),
        }
        (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Keep rules:
    # 1) Do not modify row['text'], audio, prompt.
    # 2) Do not auto-fill semantics when semantics == [].
    # 3) Do not auto-change domain/intent/invalid slot names.
    # 4) Do not force vehicle-control domain for 車載交互位置/車載交互設備 slots in other legal domains.
    # 5) Do not treat normalized placeholders (request/confirm/true/max/min/all/+/-/asc) as errors by default.
    main()
