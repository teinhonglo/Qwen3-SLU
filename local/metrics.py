import json
import argparse
import sys
import re
import os
import traceback

def normalize_text(text):
    if not isinstance(text, str):
        return str(text)
    text = text.lower()
    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '两': '2',
        '幺': '1'
    }
    for k, v in cn_num_map.items():
        text = text.replace(k, v)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def normalize_semantics(semantics_list):
    if not isinstance(semantics_list, list):
        return []
    normalized_list = []
    for item in semantics_list:
        if not isinstance(item, dict):
            continue
        new_item = {}
        if 'domain' in item:
            new_item['domain'] = normalize_text(item['domain'])
        if 'intent' in item:
            new_item['intent'] = normalize_text(item['intent'])

        if 'slots' in item:
            origin_slots = item['slots']
            if isinstance(origin_slots, dict):
                new_slots = {}
                for k, v in origin_slots.items():
                    new_slots[normalize_text(k)] = normalize_text(v)
                new_item['slots'] = new_slots
            else:
                new_item['slots'] = origin_slots

        if 'implicit_slots' in item:
            origin_slots = item['implicit_slots']
            if isinstance(origin_slots, dict):
                new_slots = {}
                for k, v in origin_slots.items():
                    new_slots[normalize_text(k)] = normalize_text(v)
                new_item['implicit_slots'] = new_slots
            else:
                new_item['implicit_slots'] = origin_slots

        normalized_list.append(new_item)
    return normalized_list


def tokenize_for_mer(text):
    norm = normalize_text(text)
    tokens = []
    en_buffer = []

    def flush_en_buffer():
        if en_buffer:
            tokens.append("".join(en_buffer))
            en_buffer.clear()

    for ch in norm:
        if ch.isspace():
            flush_en_buffer()
            continue

        if '\u4e00' <= ch <= '\u9fff':
            flush_en_buffer()
            tokens.append(ch)
            continue

        if ch.isascii() and (ch.isalnum() or ch == "'"):
            en_buffer.append(ch)
            continue

        flush_en_buffer()
        tokens.append(ch)

    flush_en_buffer()
    return [t for t in tokens if t]


def edit_distance(ref_tokens, hyp_tokens):
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[m]

def slot_mer_metric(slot_tp, slot_fp, slot_fn):
    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) else (1.0 if slot_fn == 0 else 0.0)
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) else (1.0 if slot_fp == 0 else 0.0)
    slot_f1 = 0.0 if (slot_precision + slot_recall) == 0 else 2 * slot_precision * slot_recall / (slot_precision + slot_recall)

    return slot_precision, slot_recall, slot_f1


def get_intent_group(intent_num):
    if intent_num == 1:
        return "1_intent"
    if intent_num == 2:
        return "2_intent"
    if intent_num >= 3:
        return "3plus_intent"
    return None


def init_group_stats():
    return {
        "total_count": 0,
        "overall_match_count": 0,
        "intent_match_count": 0,
        "slot_tp": 0,
        "slot_fp": 0,
        "slot_fn": 0,
        "implicit_slot_tp": 0,
        "implicit_slot_fp": 0,
        "implicit_slot_fn": 0,
        "query_mer_errors": 0,
        "query_mer_ref_lens": 0,
        "slot_match_counts": 0,
        "valid_slotss": 0,
    }


def finalize_group_stats(group_stats):
    slot_precision, slot_recall, slot_f1 = slot_mer_metric(
        group_stats["slot_tp"], group_stats["slot_fp"], group_stats["slot_fn"]
    )
    implicit_slot_precision, implicit_slot_recall, implicit_slot_f1 = slot_mer_metric(
        group_stats["implicit_slot_tp"], group_stats["implicit_slot_fp"], group_stats["implicit_slot_fn"]
    )

    total_count = group_stats["total_count"]
    return {
        "total_count": total_count,
        "overall_match_count": group_stats["overall_match_count"],
        "overall_accuracy": group_stats["overall_match_count"] / total_count if total_count else 0.0,
        "intent_match_count": group_stats["intent_match_count"],
        "intent_accuracy": group_stats["intent_match_count"] / total_count if total_count else 0.0,
        "slot_tp": group_stats["slot_tp"],
        "slot_fp": group_stats["slot_fp"],
        "slot_fn": group_stats["slot_fn"],
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "implicit_slot_tp": group_stats["implicit_slot_tp"],
        "implicit_slot_fp": group_stats["implicit_slot_fp"],
        "implicit_slot_fn": group_stats["implicit_slot_fn"],
        "implicit_slot_precision": implicit_slot_precision,
        "implicit_slot_recall": implicit_slot_recall,
        "implicit_slot_f1": implicit_slot_f1,
        "query_mer_errors": group_stats["query_mer_errors"],
        "query_mer_ref_lens": group_stats["query_mer_ref_lens"],
        "query_mer": group_stats["query_mer_errors"] / group_stats["query_mer_ref_lens"] if group_stats["query_mer_ref_lens"] else 0.0,
        "slot_match_accs": group_stats["slot_match_counts"] / group_stats["valid_slotss"] if group_stats["valid_slotss"] else 0.0,
    }


def calculate_metrics(predict_file, ground_truth_file):
    with open(predict_file, 'r', encoding='utf-8') as f_pred:
        predict_lines = f_pred.readlines()
    with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        ground_truth_lines = f_gt.readlines()

    if len(predict_lines) != len(ground_truth_lines):
        print(f"Error: line count mismatch. pred={len(predict_lines)} gt={len(ground_truth_lines)}", file=sys.stderr)
        sys.exit(1)

    total_count = len(predict_lines)
    success_count =0
    overall_match_count = 0
    intent_match_count = 0
    slot_tp = slot_fp = slot_fn = 0
    implicit_slot_tp = implicit_slot_fp = implicit_slot_fn = 0
    mer_errors = 0
    mer_ref_lens = 0
    slot_match_counts = 0
    valid_slotss = 0
    intent_group_stats = {
        "1_intent": init_group_stats(),
        "2_intent": init_group_stats(),
        "3plus_intent": init_group_stats(),
    }
    report_detail = {}

    for i, (pred_line, gt_line) in enumerate(zip(predict_lines, ground_truth_lines), start=1):
        try:
            pred_data = json.loads(pred_line.strip())
            gt_data = json.loads(gt_line.strip())
            
            text_id = gt_data["text_id"]

            pred_semantics = normalize_semantics(pred_data.get("pred_semantics", []))
            gt_semantics = normalize_semantics(gt_data.get("semantics", []))
            intent_group = get_intent_group(len(gt_semantics))

            report_detail[text_id] = {
                "intent_group": intent_group,
                "query_ori": gt_data.get("query", ""),
                "query": normalize_text(gt_data.get("query", "")),
                "gt_semantics_ori": gt_data.get("semantics", []),
                "gt_semantics": gt_semantics,
                "pred_query_ori": pred_data.get("pred_query", ""),
                "pred_query": normalize_text(pred_data.get("pred_query", "")),
                "pred_semantics_ori": pred_data.get("pred_semantics", []),
                "pred_semantics": pred_semantics,
                "pred_slots_ori": "",
                "intent_match": 0,
                "slot_tp": 0,
                "slot_fp": 0,
                "slot_fn": 0,
                "slot_precision": 0,
                "slot_recall": 0,
                "slot_f1": 0,
                "implicit_slot_tp": 0,
                "implicit_slot_fp": 0,
                "implicit_slot_fn": 0,
                "implicit_slot_precision": 0,
                "implicit_slot_recall": 0,
                "implicit_slot_f1": 0,
                "mer": 0
            }

            if pred_semantics == gt_semantics:
                overall_match_count += 1
                if intent_group is not None:
                    intent_group_stats[intent_group]["overall_match_count"] += 1

            pred_intents = sorted([(s.get("domain"), s.get("intent")) for s in pred_semantics])
            gt_intents = sorted([(s.get("domain"), s.get("intent")) for s in gt_semantics])
            if pred_intents == gt_intents:
                intent_match_count += 1
                report_detail[text_id]["intent_match"] = 1
                if intent_group is not None:
                    intent_group_stats[intent_group]["intent_match_count"] += 1

            # slot
            pred_slot_set = set()
            for s in pred_semantics:
                slots = s.get("slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        pred_slot_set.add((k, v))

            gt_slot_set = set()
            for s in gt_semantics:
                slots = s.get("slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        gt_slot_set.add((k, v))

            report_detail[text_id]["slot_tp"] = len(pred_slot_set & gt_slot_set)
            report_detail[text_id]["slot_fp"] = len(pred_slot_set - gt_slot_set)
            report_detail[text_id]["slot_fn"] = len(gt_slot_set - pred_slot_set)

            slot_tp += report_detail[text_id]["slot_tp"]
            slot_fp += report_detail[text_id]["slot_fp"]
            slot_fn += report_detail[text_id]["slot_fn"]
            if intent_group is not None:
                intent_group_stats[intent_group]["slot_tp"] += report_detail[text_id]["slot_tp"]
                intent_group_stats[intent_group]["slot_fp"] += report_detail[text_id]["slot_fp"]
                intent_group_stats[intent_group]["slot_fn"] += report_detail[text_id]["slot_fn"]

            # implicit slot
            pred_implicit_slot_set = set()
            for s in pred_semantics:
                slots = s.get("implicit_slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        pred_implicit_slot_set.add((k, v))

            gt_implicit_slot_set = set()
            for s in gt_semantics:
                slots = s.get("implicit_slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        gt_implicit_slot_set.add((k, v))

            report_detail[text_id]["implicit_slot_tp"] = len(pred_implicit_slot_set & gt_implicit_slot_set)
            report_detail[text_id]["implicit_slot_fp"] = len(pred_implicit_slot_set - gt_implicit_slot_set)
            report_detail[text_id]["implicit_slot_fn"] = len(gt_implicit_slot_set - pred_implicit_slot_set)

            implicit_slot_tp += report_detail[text_id]["implicit_slot_tp"]
            implicit_slot_fp += report_detail[text_id]["implicit_slot_fp"]
            implicit_slot_fn += report_detail[text_id]["implicit_slot_fn"]
            if intent_group is not None:
                intent_group_stats[intent_group]["implicit_slot_tp"] += report_detail[text_id]["implicit_slot_tp"]
                intent_group_stats[intent_group]["implicit_slot_fp"] += report_detail[text_id]["implicit_slot_fp"]
                intent_group_stats[intent_group]["implicit_slot_fn"] += report_detail[text_id]["implicit_slot_fn"]

            # MER
            query_ref_tokens = tokenize_for_mer(report_detail[text_id]["query"])
            query_hyp_tokens = tokenize_for_mer(report_detail[text_id]["pred_query"])

            mer_error = edit_distance(query_ref_tokens, query_hyp_tokens)
            mer_ref_len = len(query_ref_tokens)
            mer = mer_error / mer_ref_len if mer_ref_len else 0.0

            mer_errors += mer_error
            mer_ref_lens += mer_ref_len
            if intent_group is not None:
                intent_group_stats[intent_group]["query_mer_errors"] += mer_error
                intent_group_stats[intent_group]["query_mer_ref_lens"] += mer_ref_len

            # original slot (match)
            pred_slots_value = []
            for s in report_detail[text_id]["pred_semantics_ori"]:
                slots = s.get("slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        pred_slots_value.append(v)

            report_detail[text_id]["pred_slots_ori"] = pred_slots_value
            slot_match_count = 0
            valid_slots = len(report_detail[text_id]["pred_slots_ori"])
            
            for _slot in report_detail[text_id]["pred_slots_ori"]:
                if _slot in report_detail[text_id]["pred_query_ori"]:
                    slot_match_count += 1
                else:
                    if _slot in str(report_detail[text_id]["gt_semantics_ori"]):
                        valid_slots -= 1

            slot_match_counts += slot_match_count
            valid_slotss += valid_slots
            if intent_group is not None:
                intent_group_stats[intent_group]["slot_match_counts"] += slot_match_count
                intent_group_stats[intent_group]["valid_slotss"] += valid_slots
            slot_match_acc = slot_match_count / valid_slots if valid_slots else 0.0
            
            success_count += 1
            if intent_group is not None:
                intent_group_stats[intent_group]["total_count"] += 1

            slot_precision, slot_recall, slot_f1 = slot_mer_metric(report_detail[text_id]["slot_tp"], report_detail[text_id]["slot_fp"], report_detail[text_id]["slot_fn"])

            report_detail[text_id]["slot_precision"] = slot_precision
            report_detail[text_id]["slot_recall"] = slot_recall
            report_detail[text_id]["slot_f1"] = slot_f1
            report_detail[text_id]["mer"] = mer
            report_detail[text_id]["slot_acc"] = slot_match_acc

        except Exception as e:
            print(f"Warning: failed at line {i}: {e}", file=sys.stderr)
            traceback.print_exc()

    overall_accuracy = overall_match_count / total_count if total_count else 0.0
    intent_accuracy = intent_match_count / total_count if total_count else 0.0
    # slot
    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) else (1.0 if slot_fn == 0 else 0.0)
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) else (1.0 if slot_fp == 0 else 0.0)
    slot_f1 = 0.0 if (slot_precision + slot_recall) == 0 else 2 * slot_precision * slot_recall / (slot_precision + slot_recall)
    implicit_slot_precision = implicit_slot_tp / (implicit_slot_tp + implicit_slot_fp) if (implicit_slot_tp + implicit_slot_fp) else (1.0 if implicit_slot_fn == 0 else 0.0)
    implicit_slot_recall = implicit_slot_tp / (implicit_slot_tp + implicit_slot_fn) if (implicit_slot_tp + implicit_slot_fn) else (1.0 if implicit_slot_fp == 0 else 0.0)
    implicit_slot_f1 = 0.0 if (implicit_slot_precision + implicit_slot_recall) == 0 else 2 * implicit_slot_precision * implicit_slot_recall / (implicit_slot_precision + implicit_slot_recall)
    # mer
    mer = mer_errors / mer_ref_lens if mer_ref_lens else 0.0
    slot_match_accs = slot_match_counts / valid_slotss if valid_slotss else 0.0
    intent_group_metrics = {k: finalize_group_stats(v) for k, v in intent_group_stats.items()}

    return {
        "total_count": total_count,
        "success_count": success_count,
        "overall_match_count": overall_match_count,
        "overall_accuracy": overall_accuracy,
        "intent_match_count": intent_match_count,
        "intent_accuracy": intent_accuracy,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "implicit_slot_tp": implicit_slot_tp,
        "implicit_slot_fp": implicit_slot_fp,
        "implicit_slot_fn": implicit_slot_fn,
        "implicit_slot_precision": implicit_slot_precision,
        "implicit_slot_recall": implicit_slot_recall,
        "implicit_slot_f1": implicit_slot_f1,
        "query_mer_errors": mer_errors,
        "query_mer_ref_lens": mer_ref_lens,
        "query_mer": mer,
        "slot_match_accs": slot_match_accs,
        "intent_group_metrics": intent_group_metrics
    }, report_detail


def main():
    parser = argparse.ArgumentParser(description="Calculate NLU Evaluation Metrics")
    parser.add_argument("predict_file")
    parser.add_argument("ground_truth_file")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    r, r_d = calculate_metrics(args.predict_file, args.ground_truth_file)
    print("-" * 60)
    print("Evaluation Results")
    print("-" * 60)
    print(f"Total: {r['total_count']}")
    print(f"Success_count: {r['success_count']}")
    print(f"Overall accuracy: {r['overall_accuracy']:.4f}")
    print(f"Intent accuracy:  {r['intent_accuracy']:.4f}")
    print(f"Slot P/R/F1:      {r['slot_precision']:.4f} / {r['slot_recall']:.4f} / {r['slot_f1']:.4f}")
    print(f"Implicit Slot P/R/F1:      {r['implicit_slot_precision']:.4f} / {r['implicit_slot_recall']:.4f} / {r['implicit_slot_f1']:.4f}")
    print(f"Query MER:        {r['query_mer']:.4f} ({r['query_mer_errors']}/{r['query_mer_ref_lens']})")
    print(f"Slot Match accuracy:        {r['slot_match_accs']:.4f}")
    print("-" * 60)
    for group_name, group_result in r["intent_group_metrics"].items():
        print(f"[{group_name}] Total: {group_result['total_count']}")
        print(f"[{group_name}] Overall accuracy: {group_result['overall_accuracy']:.4f}")
        print(f"[{group_name}] Intent accuracy:  {group_result['intent_accuracy']:.4f}")
        print(f"[{group_name}] Slot P/R/F1:      {group_result['slot_precision']:.4f} / {group_result['slot_recall']:.4f} / {group_result['slot_f1']:.4f}")
        print(f"[{group_name}] Implicit Slot P/R/F1:      {group_result['implicit_slot_precision']:.4f} / {group_result['implicit_slot_recall']:.4f} / {group_result['implicit_slot_f1']:.4f}")
        print(f"[{group_name}] Query MER:        {group_result['query_mer']:.4f} ({group_result['query_mer_errors']}/{group_result['query_mer_ref_lens']})")
        print(f"[{group_name}] Slot Match accuracy:        {group_result['slot_match_accs']:.4f}")
        print("-" * 60)

    with open(os.path.join(args.output_dir, "report_details.json"), "w") as write_file:
        json.dump(r_d, write_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
