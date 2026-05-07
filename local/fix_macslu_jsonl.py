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

# Embedded schema text (default source). This is intentionally defined in-code
# so the fixer can run even when external labels files are unavailable.
DOMAIN_INTENT_LIST = """
- 车载控制
    - 车机控制
    - 车身控制
    - 提供信息
- 地图
    - 导航
    - 提供地址
    - 查询路况
    - 查询定位
    - 查询路程
    - 查询前方路线
    - 导航路线规划
    - 设置常用地址
    - 导航到常用地址
    - 沿途搜索
    - 周边搜索
    - 增加途经点
    - 删除途经点
    - 地图操作
    - 上报事件
    - sys.确认
    - sys.取消
    - sys.用户选择
    - 限速查询
    - 设置目的地
    - 查询目的地
    - 修改途经点
    - 收藏
    - 取消收藏
- 音乐
    - 播放音乐
    - 播放控制
    - 查询音乐信息
    - 播放收藏
    - 播放列表
    - 播放历史
    - 新手引导
    - sys.用户选择
    - sys.确认
    - sys.取消
- 打电话
    - 拨打电话
    - 电话控制
    - 接听电话
    - 挂断电话
    - sys.确认
    - sys.取消
    - 查询信息
    - sys.电话选择
    - 拨打黄页号码
- 收音机
    - 播放电台
    - 播放控制
    - 播放收藏
    - 收音机控制
- 天气
    - 查询天气
    - 查询气象
    - 查询温度
    - 查询湿度
    - 查询风力
    - 查询风向
    - 查询空气质量
    - 查询紫外线
    - 查询日出日落
    - 查询活动
    - 查询装备
    - 穿衣推荐
    - 新手引导
    - 查询日期
    - 查询城市
    - 查询场景
    - 查询护肤品
    - 查询能见度
    - 查询指数
    - 查询降水量
    - 查询降雪量
    - sys.确认
    - sys.取消
    - sys.用户选择
- 影视
    - 播放影视
    - 播放控制
    - 播放收藏
    - 播放列表
    - 播放历史
    - sys.确认
    - sys.取消
    - sys.用户选择
    - 查询影视信息
- 播放控制
    - 播放控制
"""

SLOT_LIST = """
- 地图-__act__:
- 地图-__tgt__:
- 地图-poi修饰:
- 地图-poi名称:
- 地图-poi目标:
- 地图-poi类型:
- 地图-sys.序列号:
- 地图-sys.指代:
- 地图-sys.页码:
- 地图-事件:
- 地图-充电功率:
- 地图-充电品牌:
- 地图-地图尺寸:
- 地图-对象:
- 地图-导航视角:
- 地图-导航道路位置:
- 地图-操作:
- 地图-模式:
- 地图-电站筛选条件:
- 地图-终点修饰:
- 地图-终点名称:
- 地图-终点目标:
- 地图-终点类型:
- 地图-请求类型:
- 地图-起点修饰:
- 地图-起点名称:
- 地图-起点类型:
- 地图-距离:
- 地图-距离排序:
- 地图-路线偏好:
- 地图-车载交互位置:
- 地图-车载交互设备:
- 地图-途经点修饰:
- 地图-途经点名称:
- 地图-途经点目标:
- 地图-途经点类型:
- 天气-__act__:
- 天气-__tgt__:
- 影视-片名:
- 收音机-频道:
- 车载控制-value:
- 车载控制-调节内容:
- 车载控制-车内灯类型:
- 音乐-歌曲名:
- 音乐-歌手名:
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--write_reports", action="store_true")
    return p.parse_args()


def load_labels_schema(labels_path: Path):
    raw_text = labels_path.read_text(encoding="utf-8") if labels_path.exists() else ""
    lines = raw_text.splitlines()
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

    # Fallback #1: support python-constant style schema text, e.g.
    # DOMAIN_INTENT_LIST = """...""" and SLOT_LIST = """..."""
    if not valid_domains or not valid_intents_by_domain:
        di_match = re.search(r"DOMAIN_INTENT_LIST\s*=\s*\"\"\"(.*?)\"\"\"", raw_text, flags=re.S)
        sl_match = re.search(r"SLOT_LIST\s*=\s*\"\"\"(.*?)\"\"\"", raw_text, flags=re.S)
        if di_match:
            current_domain = None
            for raw in di_match.group(1).splitlines():
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue
                if line.startswith("- "):
                    current_domain = stripped[2:].strip()
                    valid_domains.add(current_domain)
                    valid_intents_by_domain[current_domain]
                elif line.startswith("    - ") and current_domain:
                    valid_intents_by_domain[current_domain].add(stripped[2:].strip())
        if sl_match:
            for raw in sl_match.group(1).splitlines():
                stripped = raw.strip()
                if not stripped.startswith("- "):
                    continue
                left = stripped[2:].split(":", 1)[0].strip()
                if "-" not in left:
                    continue
                domain, slot = left.split("-", 1)
                domain, slot = domain.strip(), slot.strip()
                if domain:
                    valid_domains.add(domain)
                    valid_slots_by_domain[domain].add(slot)

    # Fallback #2: use embedded in-code schema constants.
    if not valid_domains or not valid_intents_by_domain:
        current_domain = None
        for raw in DOMAIN_INTENT_LIST.splitlines():
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if line.startswith("- "):
                current_domain = stripped[2:].strip()
                valid_domains.add(current_domain)
                valid_intents_by_domain[current_domain]
            elif line.startswith("    - ") and current_domain:
                valid_intents_by_domain[current_domain].add(stripped[2:].strip())
        for raw in SLOT_LIST.splitlines():
            stripped = raw.strip()
            if not stripped.startswith("- "):
                continue
            left = stripped[2:].split(":", 1)[0].strip()
            if "-" not in left:
                continue
            domain, slot = left.split("-", 1)
            domain, slot = domain.strip(), slot.strip()
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
