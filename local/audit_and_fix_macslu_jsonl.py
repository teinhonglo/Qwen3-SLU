#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import re
from pathlib import Path

PREFIX = "language None<asr_text>"

LEGACY_SLOT_MAPPING = {
    "body": "对象",
    "object": "对象",
    "feature": "对象功能",
    "part": "调节内容",
    "action": "操作"
}

CAR_BODY_OBJECTS = {
    "空调", "座椅", "车窗", "窗户", "天窗",
    "大灯", "远光灯", "近光灯", "雾灯", "灯",
    "遮阳帘", "后视镜", "方向盘", "氛围灯", "扶手台"
}

CAR_MACHINE_OBJECTS = {
    "屏幕", "显示屏", "扬声器", "蓝牙", "电话", "多媒体"
}

CAR_FUNCTIONS = {
    "通风", "按摩", "加热", "除霜", "除雾",
    "制冷", "制热", "内循环", "外循环",
    "自然风", "声浪模拟"
}

PLAYBACK_DOMAINS = {"音乐", "影视"}
PLAYBACK_INTENTS = {"播放音乐", "播放影视", "播放控制"}
PLACEHOLDER_CANDIDATES = {
    "all": ["全部", "所有", "全车"],
    "mid": ["中", "中间", "平衡点"],
    "max": ["最大", "最高"],
    "min": ["最小", "最低"]
}
DEPENDENT_FUNCTIONS = {"通风", "按摩", "加热", "除霜", "除雾"}
TARGET_SLOT_NAMES = {"value", "模式", "页面", "位置", "座椅记忆位置"}
VALUE_PATTERNS = [
    r"[一二三四五六七八九十两零〇百千万]+度",
    r"[一二三四五六七八九十两零〇百千万]+挡",
    r"\d+度",
    r"\d+挡",
    r"最大",
    r"最小",
    r"最高",
    r"最低",
    r"自动",
    r"中",
    r"玫红色|红色|蓝色|绿色|白色|黑色|黄色|紫色|橙色"
]
DOMAIN_TRIGGERS = {
    "车载控制": ["打开", "关闭", "调到", "设置", "切换", "空调", "车窗", "座椅", "天窗", "灯", "屏幕"],
    "音乐": ["播放", "放一首", "听", "歌", "音乐", "歌手", "歌曲"],
    "地图": ["导航", "去", "附近", "停车场", "地址", "路线", "路况", "回家", "公司"],
    "打电话": ["打电话", "拨打", "接电话", "挂断", "联系人", "号码"],
    "天气": ["天气", "温度", "下雨", "冷", "热", "风力", "湿度", "空气质量"],
    "影视": ["播放", "看", "电影", "电视剧", "综艺", "演员", "导演"]
}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}") from e
    return rows


def write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_text_field(row):
    semantics_text = json.dumps(row["semantics"], ensure_ascii=False)
    payload = {"asr_text": row["query"], "semantics": semantics_text}
    row["text"] = PREFIX + json.dumps(payload, ensure_ascii=False)


def extract_text_payload(row):
    text = row.get("text", "")
    if not isinstance(text, str) or not text.startswith(PREFIX):
        return None
    try:
        return json.loads(text[len(PREFIX):])
    except Exception:
        return None


def check_text_consistency(row):
    payload = extract_text_payload(row)
    if payload is None:
        return False
    if payload.get("asr_text") != row.get("query"):
        return False
    try:
        sem = json.loads(payload.get("semantics", ""))
    except Exception:
        return False
    return sem == row.get("semantics")


def get_all_slot_pairs(frame):
    merged = {}
    if isinstance(frame.get("slots"), dict):
        merged.update(frame["slots"])
    if isinstance(frame.get("implicit_slots"), dict):
        merged.update(frame["implicit_slots"])
    return list(merged.items())


def rebuild_slots(frame, query):
    merged = {}
    if isinstance(frame.get("slots"), dict):
        merged.update(frame["slots"])
    if isinstance(frame.get("implicit_slots"), dict):
        merged.update(frame["implicit_slots"])
    slots, implicit_slots = {}, {}
    for k, v in merged.items():
        if v is None or str(v) == "":
            continue
        if str(v) in query:
            slots[k] = v
        else:
            implicit_slots[k] = v
    frame["slots"] = slots
    frame["implicit_slots"] = implicit_slots
    return frame


def record_issue(issues, split, text_id, query, frame_idx, rule_id, severity, issue_type, old_frame, new_frame, fix_action, need_manual_review, note=""):
    issues.append({
        "split": split, "text_id": text_id, "query": query, "frame_idx": frame_idx,
        "rule_id": rule_id, "severity": severity, "issue_type": issue_type,
        "old_frame": json.dumps(old_frame, ensure_ascii=False),
        "new_frame": json.dumps(new_frame, ensure_ascii=False),
        "fix_action": fix_action, "need_manual_review": str(bool(need_manual_review)).lower(), "note": note
    })


def record_manual_review(manual_reviews, split, text_id, query, frame_idx, rule_id, severity, issue_type, frame, note=""):
    manual_reviews.append({
        "split": split, "text_id": text_id, "query": query, "frame_idx": frame_idx,
        "rule_id": rule_id, "severity": severity, "issue_type": issue_type,
        "frame": json.dumps(frame, ensure_ascii=False), "note": note
    })


def audit_and_fix_frame(frame, query, split, text_id, frame_idx, issues, manual_reviews):
    frame = copy.deepcopy(frame) if isinstance(frame, dict) else {}
    frame.setdefault("domain", "")
    frame.setdefault("intent", "")
    frame["slots"] = frame.get("slots") if isinstance(frame.get("slots"), dict) else {}
    frame["implicit_slots"] = frame.get("implicit_slots") if isinstance(frame.get("implicit_slots"), dict) else {}

    def apply_change(rule_id, severity, issue_type, old, action, note=""):
        record_issue(issues, split, text_id, query, frame_idx, rule_id, severity, issue_type, old, frame, action, False, note)

    # R1
    old = copy.deepcopy(frame)
    changed = False
    for src, dst in LEGACY_SLOT_MAPPING.items():
        for bucket in ("slots", "implicit_slots"):
            if src in frame[bucket]:
                val = frame[bucket].pop(src)
                frame[bucket][dst] = val
                changed = True
    if changed:
        frame = rebuild_slots(frame, query)
        apply_change("R1", "major", "LEGACY_SLOT_NAME", old, "map legacy slot names and rebuild")

    # R4
    old = copy.deepcopy(frame)
    merged = dict(get_all_slot_pairs(frame))
    if merged.get("操作_concrete") == "true":
        replacement = None
        for c in ["到", "为", "成", "至"]:
            if c in query:
                replacement = c
                break
        if replacement is None:
            del merged["操作_concrete"]
        else:
            merged["操作_concrete"] = replacement
        frame["slots"], frame["implicit_slots"] = merged, {}
        frame = rebuild_slots(frame, query)
        apply_change("R4", "major", "INVALID_OPERATION_CONCRETE_TRUE", old, "replace/delete 操作_concrete=true and rebuild")

    # R5/R6/R8
    merged = dict(get_all_slot_pairs(frame))
    for k in list(merged.keys()):
        v = merged[k]
        # R5
        if isinstance(v, str) and v in PLACEHOLDER_CANDIDATES:
            oldf = copy.deepcopy(frame)
            found = None
            for cand in PLACEHOLDER_CANDIDATES[v]:
                if cand in query:
                    found = cand
                    break
            if found is not None:
                merged[k] = found
                frame["slots"], frame["implicit_slots"] = merged, {}
                frame = rebuild_slots(frame, query)
                merged = dict(get_all_slot_pairs(frame))
                apply_change("R5", "major", "PLACEHOLDER_VALUE", oldf, f"replace {v} -> {found} and rebuild")
            else:
                record_manual_review(manual_reviews, split, text_id, query, frame_idx, "R5", "major", "PLACEHOLDER_VALUE_NEEDS_REVIEW", frame, f"No candidate found for {v}")
                record_issue(issues, split, text_id, query, frame_idx, "R5", "major", "PLACEHOLDER_VALUE", oldf, frame, "no auto-fix", True, "No candidate found")
        # R6
        if k in {"value", "模式", "位置", "座椅记忆位置"} and isinstance(v, str):
            for suffix in ["度", "挡", "级", "%"]:
                ext = v + suffix
                if ext in query and ext != v:
                    oldf = copy.deepcopy(frame)
                    merged[k] = ext
                    frame["slots"], frame["implicit_slots"] = merged, {}
                    frame = rebuild_slots(frame, query)
                    merged = dict(get_all_slot_pairs(frame))
                    apply_change("R6", "minor", "VALUE_SURFACE_UNIT_EXPANSION", oldf, f"expand value to {ext} and rebuild")
                    break

    # R8
    merged = dict(get_all_slot_pairs(frame))
    if "功能" in merged and merged["功能"] in DEPENDENT_FUNCTIONS:
        val = merged["功能"]
        oldf = copy.deepcopy(frame)
        if "座椅" in query and val in {"通风", "按摩", "加热"}:
            del merged["功能"]
            merged["对象"] = "座椅"
            merged["对象功能"] = val
            frame["slots"], frame["implicit_slots"] = merged, {}
            frame = rebuild_slots(frame, query)
            apply_change("R8", "major", "FUNCTION_SHOULD_BE_OBJECT_FUNCTION", oldf, "convert 功能->对象/对象功能 for 座椅")
        elif "空调" in query and val in {"除霜", "除雾"}:
            del merged["功能"]
            merged["对象"] = "空调"
            merged["对象功能"] = val
            frame["slots"], frame["implicit_slots"] = merged, {}
            frame = rebuild_slots(frame, query)
            apply_change("R8", "major", "FUNCTION_SHOULD_BE_OBJECT_FUNCTION", oldf, "convert 功能->对象/对象功能 for 空调")
        else:
            record_manual_review(manual_reviews, split, text_id, query, frame_idx, "R8", "major", "FUNCTION_OBJECT_NEEDS_REVIEW", frame, "Cannot infer object")
            record_issue(issues, split, text_id, query, frame_idx, "R8", "major", "FUNCTION_SHOULD_BE_OBJECT_FUNCTION", oldf, frame, "no auto-fix", True, "Cannot infer object")

    # R2/R3
    values = [str(v) for _, v in get_all_slot_pairs(frame)]
    oldf = copy.deepcopy(frame)
    changed = False
    for v in values:
        if v in CAR_MACHINE_OBJECTS and frame.get("domain") != "车载控制":
            frame["domain"], frame["intent"] = "车载控制", "车机控制"
            changed = True
            break
        if (v in CAR_BODY_OBJECTS or v in CAR_FUNCTIONS) and frame.get("domain") != "车载控制":
            frame["domain"], frame["intent"] = "车载控制", "车身控制"
            changed = True
            break
    if changed:
        apply_change("R2", "critical", "CAR_CONTROL_SLOT_WRONG_DOMAIN", oldf, "fix domain/intent based on car-control slot")

    oldf = copy.deepcopy(frame)
    if frame.get("domain") in PLAYBACK_DOMAINS and frame.get("intent") in PLAYBACK_INTENTS:
        for v in values:
            if v in CAR_MACHINE_OBJECTS:
                frame["domain"], frame["intent"] = "车载控制", "车机控制"
                apply_change("R3", "critical", "PLAYBACK_INTENT_WITH_CAR_CONTROL_SLOT", oldf, "fix to car-machine control")
                break
            if v in CAR_BODY_OBJECTS or v in CAR_FUNCTIONS:
                frame["domain"], frame["intent"] = "车载控制", "车身控制"
                apply_change("R3", "critical", "PLAYBACK_INTENT_WITH_CAR_CONTROL_SLOT", oldf, "fix to car-body control")
                break

    # R7
    merged = dict(get_all_slot_pairs(frame))
    is_adjust = ("操作_concrete" in merged) or any(x in query for x in ["调到", "设为", "设置为", "开到", "调为", "设成"])
    has_target = any(k in merged for k in TARGET_SLOT_NAMES)
    if is_adjust and not has_target:
        oldf = copy.deepcopy(frame)
        candidates = []
        for p in VALUE_PATTERNS:
            candidates.extend(re.findall(p, query))
        uniq = []
        for c in candidates:
            if c not in uniq:
                uniq.append(c)
        if len(uniq) == 1:
            merged["value"] = uniq[0]
            frame["slots"], frame["implicit_slots"] = merged, {}
            frame = rebuild_slots(frame, query)
            apply_change("R7", "major", "MISSING_TARGET_VALUE", oldf, f"add value={uniq[0]} and rebuild")
        else:
            record_manual_review(manual_reviews, split, text_id, query, frame_idx, "R7", "major", "MISSING_TARGET_VALUE_NEEDS_REVIEW", frame, f"candidates={uniq}")
            record_issue(issues, split, text_id, query, frame_idx, "R7", "major", "MISSING_TARGET_VALUE", oldf, frame, "no auto-fix", True, f"candidates={uniq}")

    # R9
    if not frame.get("domain") or not frame.get("intent"):
        record_manual_review(manual_reviews, split, text_id, query, frame_idx, "R9", "critical", "EMPTY_DOMAIN_OR_INTENT", frame, "Domain or intent empty")
        record_issue(issues, split, text_id, query, frame_idx, "R9", "critical", "EMPTY_DOMAIN_OR_INTENT", frame, frame, "no auto-fix", True, "Domain or intent empty")

    # R11
    oldf = copy.deepcopy(frame)
    rebuilt = rebuild_slots(frame, query)
    if rebuilt != oldf:
        frame = rebuilt
        apply_change("R11", "major", "SLOT_IMPLICIT_INVARIANT_REBUILD", oldf, "final invariant rebuild")
    return frame


def audit_and_fix_row(row, split, issues, manual_reviews):
    row = copy.deepcopy(row)
    text_id = row.get("text_id", "")
    query = row.get("query", "")
    semantics = row.get("semantics", []) if isinstance(row.get("semantics", []), list) else []

    if len(semantics) == 0:
        for _, triggers in DOMAIN_TRIGGERS.items():
            if any(t in query for t in triggers):
                record_manual_review(manual_reviews, split, text_id, query, -1, "R10", "critical", "EMPTY_SEMANTICS_WITH_OBVIOUS_INTENT", {"semantics": semantics}, "Semantics empty but query has obvious triggers")
                record_issue(issues, split, text_id, query, -1, "R10", "critical", "EMPTY_SEMANTICS_WITH_OBVIOUS_INTENT", {"semantics": semantics}, {"semantics": semantics}, "no auto-fix", True, "Semantics empty with trigger")
                break

    fixed_semantics = []
    for idx, frame in enumerate(semantics):
        fixed_semantics.append(audit_and_fix_frame(frame, query, split, text_id, idx, issues, manual_reviews))
    row["semantics"] = fixed_semantics

    old_text = row.get("text", "")
    consistent = check_text_consistency(row)
    update_text_field(row)
    if (not consistent) or old_text != row["text"]:
        record_issue(issues, split, text_id, query, -1, "R12", "critical", "TEXT_FIELD_INCONSISTENT", {"text": old_text}, {"text": row["text"]}, "rebuild text payload", False, "Text field synchronized")
    return row


def run_tests():
    def process(row):
        issues, manual = [], []
        out = audit_and_fix_row(row, "test", issues, manual)
        return out

    r1 = {"text_id":"id_test_1","query":"播放音乐打开空调","audio":"dummy.wav","prompt":"dummy","text":"","semantics":[{"domain":"音乐","intent":"播放音乐","slots":{"操作":"打开","对象":"空调"},"implicit_slots":{}}]}
    o1 = process(r1)
    assert o1["semantics"][0]["domain"] == "车载控制" and o1["semantics"][0]["intent"] == "车身控制"

    r2 = {"text_id":"id2","query":"把空调调到合适的温度","audio":"a","prompt":"p","text":"","semantics":[{"domain":"车载控制","intent":"车身控制","slots":{"对象":"空调","操作":"调","操作_concrete":"true"},"implicit_slots":{}}]}
    o2 = process(r2)
    assert o2["semantics"][0]["slots"].get("操作_concrete") == "到"

    r3 = {"text_id":"id3","query":"关闭所有车窗","audio":"a","prompt":"p","text":"","semantics":[{"domain":"车载控制","intent":"车身控制","slots":{"位置":"all","对象":"车窗"},"implicit_slots":{}}]}
    o3 = process(r3)
    assert o3["semantics"][0]["slots"].get("位置") == "所有"

    r4 = {"text_id":"id4","query":"空调温度设为二十三度","audio":"a","prompt":"p","text":"","semantics":[{"domain":"车载控制","intent":"车身控制","slots":{"对象":"空调","调节内容":"温度","操作":"设","操作_concrete":"为","value":"二十三"},"implicit_slots":{}}]}
    o4 = process(r4)
    assert o4["semantics"][0]["slots"].get("value") == "二十三度"

    r5 = {"text_id":"id5","query":"空调风量开到四挡","audio":"a","prompt":"p","text":"","semantics":[{"domain":"车载控制","intent":"车身控制","slots":{"对象":"空调","调节内容":"风量","操作":"开","操作_concrete":"到"},"implicit_slots":{}}]}
    o5 = process(r5)
    assert o5["semantics"][0]["slots"].get("value") == "四挡"

    r6 = {"text_id":"id6","query":"玻璃起雾了","audio":"a","prompt":"p","text":"","semantics":[{"domain":"车载控制","intent":"车身控制","slots":{"对象":"空调","对象功能":"除霜","操作":"打开"},"implicit_slots":{}}]}
    o6 = process(r6)
    assert o6["semantics"][0]["slots"] == {} and o6["semantics"][0]["implicit_slots"].get("对象") == "空调"

    old_payload = {"asr_text":"播放音乐打开空调","semantics":json.dumps([{"domain":"音乐","intent":"播放音乐","slots":{"操作":"打开","对象":"空调"},"implicit_slots":{}}], ensure_ascii=False)}
    r7 = {"text_id":"id7","query":"播放音乐打开空调","audio":"a","prompt":"p","text":PREFIX + json.dumps(old_payload, ensure_ascii=False),"semantics":[{"domain":"音乐","intent":"播放音乐","slots":{"操作":"打开","对象":"空调"},"implicit_slots":{}}]}
    o7 = process(r7)
    p7 = extract_text_payload(o7)
    assert o7["semantics"][0]["domain"] == "车载控制"
    assert p7["asr_text"] == o7["query"] and json.loads(p7["semantics"]) == o7["semantics"]
    print("All tests passed.")


def main():
    ap = argparse.ArgumentParser(description="Audit/fix MAC-SLU JSONL and emit reports/fixed outputs")
    ap.add_argument("--jsonl-root", default="", help="Directory containing <split>.jsonl input files")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to process")
    ap.add_argument("--output-dir", default="", help="Directory to write summary.json, issues.csv, manual_review.csv and corrected <split>.jsonl")
    ap.add_argument("--run-tests", action="store_true", help="Run built-in tests and exit")
    args = ap.parse_args()

    if args.run_tests:
        run_tests()
        return

    if not args.jsonl_root or not args.output_dir:
        raise ValueError("--jsonl-root and --output-dir are required unless --run-tests")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issues, manual_reviews = [], []
    fixed_by_split = {}
    total_rows, total_frames = 0, 0

    for split in args.splits:
        rows = load_jsonl(Path(args.jsonl_root) / f"{split}.jsonl")
        fixed_rows = []
        for row in rows:
            total_rows += 1
            total_frames += len(row.get("semantics", [])) if isinstance(row.get("semantics", []), list) else 0
            fixed_rows.append(audit_and_fix_row(row, split, issues, manual_reviews))
        fixed_by_split[split] = fixed_rows

    for split, rows in fixed_by_split.items():
        write_jsonl(rows, out_dir / f"{split}.jsonl")

    issue_fields = ["split", "text_id", "query", "frame_idx", "rule_id", "severity", "issue_type", "old_frame", "new_frame", "fix_action", "need_manual_review", "note"]
    with (out_dir / "issues.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=issue_fields)
        w.writeheader()
        w.writerows(issues)

    manual_fields = ["split", "text_id", "query", "frame_idx", "rule_id", "severity", "issue_type", "frame", "note"]
    with (out_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manual_fields)
        w.writeheader()
        w.writerows(manual_reviews)

    by_split, by_rule, by_sev = {}, {}, {}
    for i in issues:
        by_split[i["split"]] = by_split.get(i["split"], 0) + 1
        by_rule[i["rule_id"]] = by_rule.get(i["rule_id"], 0) + 1
        by_sev[i["severity"]] = by_sev.get(i["severity"], 0) + 1
    summary = {
        "total_rows": total_rows,
        "total_frames": total_frames,
        "total_issues": len(issues),
        "auto_fixed_count": sum(1 for i in issues if i["need_manual_review"] == "false"),
        "manual_review_count": len(manual_reviews),
        "by_split": by_split,
        "by_rule": by_rule,
        "by_severity": by_sev,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] summary: {(out_dir / 'summary.json')}" )
    print(f"[done] issues: {(out_dir / 'issues.csv')}" )
    print(f"[done] manual review: {(out_dir / 'manual_review.csv')}" )
    for split in args.splits:
        print(f"[done] corrected jsonl: {out_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
