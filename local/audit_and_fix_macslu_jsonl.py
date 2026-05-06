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
    "action": "操作",
}

DOMAIN_TO_INTENTS = {
    "车载控制": {"车机控制", "车身控制", "提供信息"},
    "地图": {"导航", "提供地址", "查询路况", "查询定位", "查询路程", "查询前方路线", "导航路线规划", "设置常用地址", "导航到常用地址", "沿途搜索", "周边搜索", "增加途经点", "删除途经点", "地图操作", "上报事件", "sys.确认", "sys.取消", "sys.用户选择", "限速查询", "设置目的地", "查询目的地", "修改途经点", "收藏", "取消收藏"},
    "音乐": {"播放音乐", "播放控制", "查询音乐信息", "播放收藏", "播放列表", "播放历史", "新手引导", "sys.用户选择", "sys.确认", "sys.取消"},
    "打电话": {"拨打电话", "电话控制", "接听电话", "挂断电话", "sys.确认", "sys.取消", "查询信息", "sys.电话选择", "拨打黄页号码"},
    "收音机": {"播放电台", "播放控制", "播放收藏", "收音机控制"},
    "天气": {"查询天气", "查询气象", "查询温度", "查询湿度", "查询风力", "查询风向", "查询空气质量", "查询紫外线", "查询日出日落", "查询活动", "查询装备", "穿衣推荐", "新手引导", "查询日期", "查询城市", "查询场景", "查询护肤品", "查询能见度", "查询指数", "查询降水量", "查询降雪量", "sys.确认", "sys.取消", "sys.用户选择"},
    "影视": {"播放影视", "播放控制", "播放收藏", "播放列表", "播放历史", "sys.确认", "sys.取消", "sys.用户选择", "查询影视信息"},
    "播放控制": {"播放控制"},
}

# SLOT_LIST-derived conservative schema (includes shared and explicit exceptions)
COMMON_SLOTS = {"操作", "操作_concrete", "对象", "对象功能", "调节内容", "value", "模式", "页面", "位置", "座椅记忆位置", "车载交互位置", "车载交互设备"}
DOMAIN_TO_SLOTS = {
    "车载控制": COMMON_SLOTS | {"设备", "音源", "连接状态"},
    "地图": {"终点目标", "途经点", "路线偏好", "道路名", "poi类目", "poi名称", "页面", "操作", "车载交互位置", "车载交互设备"},
    "音乐": {"歌曲名", "歌手", "专辑", "风格", "榜单", "播放列表", "操作", "value", "模式", "车载交互位置", "车载交互设备"},
    "打电话": {"联系人", "号码", "黄页", "操作", "页面", "车载交互位置", "车载交互设备"},
    "收音机": {"电台", "频段", "频率", "操作", "value", "车载交互位置", "车载交互设备"},
    "天气": {"城市", "日期", "天气要素", "场景", "指数", "活动", "装备", "操作", "页面", "车载交互位置", "车载交互设备"},
    "影视": {"片名", "演员", "导演", "类型", "播放列表", "操作", "value", "模式", "车载交互位置", "车载交互设备"},
    "播放控制": {"操作", "value", "模式", "车载交互位置", "车载交互设备"},
}

PLACEHOLDER_CANDIDATES = {"all": ["全部", "所有", "全车"], "mid": ["中", "中间", "平衡点"], "max": ["最大", "最高"], "min": ["最小", "最低"]}
TARGET_SLOT_NAMES = {"value", "模式", "页面", "座椅记忆位置"}
VALUE_PATTERNS = [r"[一二三四五六七八九十两零〇百千万]+度", r"[一二三四五六七八九十两零〇百千万]+挡", r"\d+度", r"\d+挡", r"最大", r"最小", r"最高", r"最低", r"自动", r"中", r"玫红色|红色|蓝色|绿色|白色|黑色|黄色|紫色|橙色"]
DOMAIN_TRIGGERS = {
    "车载控制": ["打开", "关闭", "调到", "设置", "切换", "空调", "车窗", "座椅", "天窗", "灯", "屏幕"],
    "音乐": ["播放", "放一首", "听", "歌", "音乐", "歌手", "歌曲"],
    "地图": ["导航", "去", "附近", "停车场", "地址", "路线", "路况", "回家", "公司"],
    "打电话": ["打电话", "拨打", "接电话", "挂断", "联系人", "号码"],
    "天气": ["天气", "温度", "下雨", "冷", "热", "风力", "湿度", "空气质量"],
    "影视": ["播放", "看", "电影", "电视剧", "综艺", "演员", "导演"],
}
CAR_OBJECTS = {"空调", "座椅", "车窗", "窗户", "天窗"}
CAR_ACTIONS = {"打开", "关闭", "调", "设置", "开", "关"}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}") from e
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    if payload is None or payload.get("asr_text") != row.get("query"):
        return False
    try:
        return json.loads(payload.get("semantics", "")) == row.get("semantics")
    except Exception:
        return False


def get_all_slot_pairs(frame):
    merged = {}
    if isinstance(frame.get("slots"), dict):
        merged.update(frame["slots"])
    if isinstance(frame.get("implicit_slots"), dict):
        merged.update(frame["implicit_slots"])
    return list(merged.items())


def rebuild_slots(frame, query):
    merged = dict(get_all_slot_pairs(frame))
    slots, implicit = {}, {}
    for k, v in merged.items():
        if v is None or str(v) == "":
            continue
        if str(v) in query:
            slots[k] = v
        else:
            implicit[k] = v
    frame["slots"] = slots
    frame["implicit_slots"] = implicit
    return frame


def record_issue(issues, **kwargs):
    issues.append(kwargs)


def record_manual_review(manual_reviews, **kwargs):
    manual_reviews.append(kwargs)


def audit_and_fix_frame(frame, query, split, text_id, frame_idx, issues, manual_reviews, aggressive_fix=False):
    frame = copy.deepcopy(frame if isinstance(frame, dict) else {})
    frame.setdefault("domain", "")
    frame.setdefault("intent", "")
    frame["slots"] = frame.get("slots") if isinstance(frame.get("slots"), dict) else {}
    frame["implicit_slots"] = frame.get("implicit_slots") if isinstance(frame.get("implicit_slots"), dict) else {}

    changed = False

    # legacy mapping first
    old = copy.deepcopy(frame)
    for old_name, new_name in LEGACY_SLOT_MAPPING.items():
        for b in ("slots", "implicit_slots"):
            if old_name in frame[b]:
                frame[b][new_name] = frame[b].pop(old_name)
                changed = True
    if frame != old:
        record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R2", severity="major", issue_type="LEGACY_SLOT_NAME", old_frame=json.dumps(old, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action="legacy slot mapping", need_manual_review="false", note="")

    merged = dict(get_all_slot_pairs(frame))

    # R5 placeholder
    for k, v in list(merged.items()):
        if isinstance(v, str) and v in PLACEHOLDER_CANDIDATES:
            cand = next((x for x in PLACEHOLDER_CANDIDATES[v] if x in query), None)
            oldf = copy.deepcopy(frame)
            if cand:
                merged[k] = cand
                frame["slots"], frame["implicit_slots"] = merged, {}
                frame = rebuild_slots(frame, query)
                merged = dict(get_all_slot_pairs(frame))
                changed = True
                record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R5", severity="major", issue_type="PLACEHOLDER_SURFACE_FIX", old_frame=json.dumps(oldf, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action=f"{v}->{cand}", need_manual_review="false", note="")
            else:
                record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R5", severity="major", issue_type="PLACEHOLDER_SURFACE_FIX", frame=json.dumps(frame, ensure_ascii=False), note=f"placeholder={v} no surface candidate")

    # R6
    merged = dict(get_all_slot_pairs(frame))
    for k, v in list(merged.items()):
        if k in {"value", "模式", "位置", "座椅记忆位置"} and isinstance(v, str):
            for suffix in ["度", "挡", "级", "%"]:
                ext = v + suffix
                if ext in query and ext != v:
                    oldf = copy.deepcopy(frame)
                    merged[k] = ext
                    frame["slots"], frame["implicit_slots"] = merged, {}
                    frame = rebuild_slots(frame, query)
                    merged = dict(get_all_slot_pairs(frame))
                    changed = True
                    record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R6", severity="minor", issue_type="VALUE_UNIT_EXPANSION", old_frame=json.dumps(oldf, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action=f"expand to {ext}", need_manual_review="false", note="")
                    break

    # R7 conservative
    merged = dict(get_all_slot_pairs(frame))
    is_adjust = ("操作_concrete" in merged) or any(x in query for x in ["调到", "设为", "设置为", "开到", "调为", "设成"])
    has_target = any(n in merged for n in TARGET_SLOT_NAMES)
    if is_adjust and ("调节内容" in merged) and (not has_target):
        oldf = copy.deepcopy(frame)
        cands = []
        for pat in VALUE_PATTERNS:
            cands.extend(re.findall(pat, query))
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        if len(uniq) == 1:
            merged["value"] = uniq[0]
            frame["slots"], frame["implicit_slots"] = merged, {}
            frame = rebuild_slots(frame, query)
            changed = True
            record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R7", severity="major", issue_type="MISSING_TARGET_VALUE", old_frame=json.dumps(oldf, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action=f"add value={uniq[0]}", need_manual_review="false", note="")
        else:
            record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R7", severity="major", issue_type="MISSING_TARGET_VALUE", frame=json.dumps(frame, ensure_ascii=False), note=f"candidates={uniq}")

    # R8 operation_concrete=true keep unless illegal domain-slot
    merged = dict(get_all_slot_pairs(frame))
    if merged.get("操作_concrete") == "true":
        domain = frame.get("domain", "")
        legal = domain in DOMAIN_TO_SLOTS and "操作_concrete" in DOMAIN_TO_SLOTS[domain]
        if domain != "车载控制" and (not legal):
            record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R8", severity="major", issue_type="OPERATION_CONCRETE_TRUE", frame=json.dumps(frame, ensure_ascii=False), note="true outside legal domain-slot schema")

    # R1 domain-intent schema check
    domain = frame.get("domain", "")
    intent = frame.get("intent", "")
    if domain not in DOMAIN_TO_INTENTS:
        record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R1", severity="critical", issue_type="DOMAIN_INTENT_SCHEMA_CHECK", frame=json.dumps(frame, ensure_ascii=False), note=f"unknown domain={domain}")
    elif intent not in DOMAIN_TO_INTENTS[domain]:
        record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R1", severity="critical", issue_type="DOMAIN_INTENT_SCHEMA_CHECK", frame=json.dumps(frame, ensure_ascii=False), note=f"illegal intent={intent} for domain={domain}")

    # R2 domain-slot schema check
    domain = frame.get("domain", "")
    legal_slots = DOMAIN_TO_SLOTS.get(domain, set())
    for sname, _ in get_all_slot_pairs(frame):
        if domain in DOMAIN_TO_SLOTS and sname not in legal_slots:
            record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R2", severity="major", issue_type="DOMAIN_SLOT_SCHEMA_CHECK", frame=json.dumps(frame, ensure_ascii=False), note=f"illegal slot {sname} for domain {domain}")

    # R9 suspicious conflict
    merged = dict(get_all_slot_pairs(frame))
    if frame.get("domain") in {"音乐", "影视"} and frame.get("intent") in {"播放音乐", "播放影视"}:
        if merged.get("对象") in CAR_OBJECTS and merged.get("操作") in CAR_ACTIONS:
            oldf = copy.deepcopy(frame)
            if aggressive_fix:
                frame["domain"] = "车载控制"
                frame["intent"] = "车身控制"
                changed = True
                record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R9", severity="critical", issue_type="SUSPICIOUS_CAR_CONTROL_DOMAIN_CONFLICT", old_frame=json.dumps(oldf, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action="aggressive domain correction", need_manual_review="false", note="")
            else:
                record_manual_review(manual_reviews, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R9", severity="critical", issue_type="SUSPICIOUS_CAR_CONTROL_DOMAIN_CONFLICT", frame=json.dumps(frame, ensure_ascii=False), note="manual review by default")

    # R3 rebuild always
    oldf = copy.deepcopy(frame)
    frame = rebuild_slots(frame, query)
    if frame != oldf or changed:
        record_issue(issues, split=split, text_id=text_id, query=query, frame_idx=frame_idx, rule_id="R3", severity="major", issue_type="REBUILD_SLOTS_IMPLICIT_SLOTS", old_frame=json.dumps(oldf, ensure_ascii=False), new_frame=json.dumps(frame, ensure_ascii=False), fix_action="rebuild by query surface", need_manual_review="false", note="")

    return frame


def audit_and_fix_row(row, split, issues, manual_reviews, aggressive_fix=False):
    new_row = copy.deepcopy(row)
    query = new_row.get("query", "")
    semantics = new_row.get("semantics", [])
    if not isinstance(semantics, list):
        semantics = []

    if semantics == []:
        for d, ts in DOMAIN_TRIGGERS.items():
            if any(t in query for t in ts):
                record_manual_review(manual_reviews, split=split, text_id=new_row.get("text_id", ""), query=query, frame_idx=-1, rule_id="R10", severity="critical", issue_type="EMPTY_SEMANTICS_WITH_OBVIOUS_TRIGGER", frame=json.dumps({"semantics": []}, ensure_ascii=False), note=f"trigger domain={d}")
                break

    fixed = []
    for idx, frame in enumerate(semantics):
        fixed.append(audit_and_fix_frame(frame, query, split, new_row.get("text_id", ""), idx, issues, manual_reviews, aggressive_fix=aggressive_fix))
    new_row["semantics"] = fixed

    old_text = new_row.get("text", "")
    consistent = check_text_consistency(new_row)
    update_text_field(new_row)
    if (not consistent) or old_text != new_row["text"]:
        record_issue(issues, split=split, text_id=new_row.get("text_id", ""), query=query, frame_idx=-1, rule_id="R4", severity="critical", issue_type="TEXT_SYNC", old_frame=json.dumps({"text": old_text}, ensure_ascii=False), new_frame=json.dumps({"text": new_row["text"]}, ensure_ascii=False), fix_action="rebuild text", need_manual_review="false", note="")
    return new_row


def run_tests():
    issues, manual = [], []
    # test1 text sync + aggressive conflict fix
    row = {"text_id": "id1", "query": "播放音乐打开空调", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "音乐", "intent": "播放音乐", "slots": {"操作": "打开", "对象": "空调"}, "implicit_slots": {}}]}
    out = audit_and_fix_row(row, "test", issues, manual, aggressive_fix=True)
    payload = extract_text_payload(out)
    assert payload and payload["asr_text"] == out["query"] and json.loads(payload["semantics"]) == out["semantics"]

    # test2 rebuild
    issues, manual = [], []
    row = {"text_id": "id2", "query": "玻璃起雾了", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "车载控制", "intent": "车身控制", "slots": {"对象": "空调", "对象功能": "除霜", "操作": "打开"}, "implicit_slots": {}}]}
    out = audit_and_fix_row(row, "test", issues, manual)
    f = out["semantics"][0]
    assert f["slots"] == {} and set(f["implicit_slots"].keys()) == {"对象", "对象功能", "操作"}

    # test3 legal non-car with 车载交互设备
    issues, manual = [], []
    row = {"text_id": "id3", "query": "在屏幕上播放音乐", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "音乐", "intent": "播放音乐", "slots": {"车载交互设备": "屏幕"}, "implicit_slots": {}}]}
    out = audit_and_fix_row(row, "test", issues, manual)
    assert out["semantics"][0]["domain"] == "音乐"

    # test4 operation_concrete true kept
    issues, manual = [], []
    row = {"text_id": "id4", "query": "把空调调true", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "车载控制", "intent": "车身控制", "slots": {"操作_concrete": "true"}, "implicit_slots": {}}]}
    out = audit_and_fix_row(row, "test", issues, manual)
    assert dict(get_all_slot_pairs(out["semantics"][0])).get("操作_concrete") == "true"

    # test5 value unit
    issues, manual = [], []
    row = {"text_id": "id5", "query": "空调温度设为二十三度", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "车载控制", "intent": "车身控制", "slots": {"调节内容": "温度", "value": "二十三"}, "implicit_slots": {}}]}
    out = audit_and_fix_row(row, "test", issues, manual)
    assert dict(get_all_slot_pairs(out["semantics"][0])).get("value") == "二十三度"

    # test6 suspicious conflict default/aggressive
    base = {"text_id": "id6", "query": "播放音乐打开空调", "audio": "a.wav", "prompt": "p", "text": "", "semantics": [{"domain": "音乐", "intent": "播放音乐", "slots": {"操作": "打开", "对象": "空调"}, "implicit_slots": {}}]}
    out1 = audit_and_fix_row(base, "test", [], [], aggressive_fix=False)
    assert out1["semantics"][0]["domain"] == "音乐"
    out2 = audit_and_fix_row(base, "test", [], [], aggressive_fix=True)
    assert out2["semantics"][0]["domain"] == "车载控制" and out2["semantics"][0]["intent"] == "车身控制"

    print("All tests passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-root", default="")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--write-fixed", action="store_true")
    ap.add_argument("--aggressive-fix", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    if args.run_tests:
        run_tests()
        return

    if not args.jsonl_root or not args.output_dir:
        raise ValueError("--jsonl-root and --output-dir are required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    issues, manual = [], []
    fixed_by_split = {}
    total_rows, total_frames = 0, 0

    for split in args.splits:
        rows = load_jsonl(Path(args.jsonl_root) / f"{split}.jsonl")
        fixed_rows = []
        for row in rows:
            total_rows += 1
            total_frames += len(row.get("semantics", [])) if isinstance(row.get("semantics", []), list) else 0
            fixed_rows.append(audit_and_fix_row(row, split, issues, manual, aggressive_fix=args.aggressive_fix))
        fixed_by_split[split] = fixed_rows

    if args.write_fixed:
        for split, rows in fixed_by_split.items():
            write_jsonl(rows, out_dir / "fixed" / f"{split}.jsonl")

    with (out_dir / "issues.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "text_id", "query", "frame_idx", "rule_id", "severity", "issue_type", "old_frame", "new_frame", "fix_action", "need_manual_review", "note"])
        w.writeheader()
        w.writerows(issues)
    with (out_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "text_id", "query", "frame_idx", "rule_id", "severity", "issue_type", "frame", "note"])
        w.writeheader()
        w.writerows(manual)

    by_split, by_rule, by_sev = {}, {}, {}
    for it in issues:
        by_split[it["split"]] = by_split.get(it["split"], 0) + 1
        by_rule[it["rule_id"]] = by_rule.get(it["rule_id"], 0) + 1
        by_sev[it["severity"]] = by_sev.get(it["severity"], 0) + 1
    summary = {
        "total_rows": total_rows,
        "total_frames": total_frames,
        "total_issues": len(issues),
        "auto_fixed_count": sum(1 for x in issues if x["need_manual_review"] == "false"),
        "manual_review_count": len(manual),
        "by_split": by_split,
        "by_rule": by_rule,
        "by_severity": by_sev,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
