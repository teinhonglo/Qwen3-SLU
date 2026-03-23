import json
import argparse
import sys
import re


def normalize_text(text):
    if not isinstance(text, str):
        return str(text)
    text = text.lower()
    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '两': '2'
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


def calculate_metrics(predict_file, ground_truth_file):
    with open(predict_file, 'r', encoding='utf-8') as f_pred:
        predict_lines = f_pred.readlines()
    with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        ground_truth_lines = f_gt.readlines()

    if len(predict_lines) != len(ground_truth_lines):
        print(f"Error: line count mismatch. pred={len(predict_lines)} gt={len(ground_truth_lines)}", file=sys.stderr)
        sys.exit(1)

    total_count = len(predict_lines)
    overall_match_count = 0
    intent_match_count = 0
    slot_tp = slot_fp = slot_fn = 0
    mer_errors = 0
    mer_ref_len = 0

    for i, (pred_line, gt_line) in enumerate(zip(predict_lines, ground_truth_lines), start=1):
        try:
            pred_data = json.loads(pred_line.strip())
            gt_data = json.loads(gt_line.strip())

            pred_semantics = normalize_semantics(pred_data.get("semantics", []))
            gt_semantics = normalize_semantics(gt_data.get("semantics", []))

            if pred_semantics == gt_semantics:
                overall_match_count += 1

            pred_intents = sorted([(s.get("domain"), s.get("intent")) for s in pred_semantics])
            gt_intents = sorted([(s.get("domain"), s.get("intent")) for s in gt_semantics])
            if pred_intents == gt_intents:
                intent_match_count += 1

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

            slot_tp += len(pred_slot_set & gt_slot_set)
            slot_fp += len(pred_slot_set - gt_slot_set)
            slot_fn += len(gt_slot_set - pred_slot_set)

            query_ref_tokens = tokenize_for_mer(gt_data.get("query", ""))
            query_hyp_tokens = tokenize_for_mer(pred_data.get("pred_query", ""))
            mer_errors += edit_distance(query_ref_tokens, query_hyp_tokens)
            mer_ref_len += len(query_ref_tokens)
        except Exception as e:
            print(f"Warning: failed at line {i}: {e}", file=sys.stderr)

    overall_accuracy = overall_match_count / total_count if total_count else 0.0
    intent_accuracy = intent_match_count / total_count if total_count else 0.0

    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) else (1.0 if slot_fn == 0 else 0.0)
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) else (1.0 if slot_fp == 0 else 0.0)
    slot_f1 = 0.0 if (slot_precision + slot_recall) == 0 else 2 * slot_precision * slot_recall / (slot_precision + slot_recall)
    mer = mer_errors / mer_ref_len if mer_ref_len else 0.0

    return {
        "total_count": total_count,
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
        "query_mer_errors": mer_errors,
        "query_mer_ref_len": mer_ref_len,
        "query_mer": mer,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate NLU Evaluation Metrics")
    parser.add_argument("predict_file")
    parser.add_argument("ground_truth_file")
    args = parser.parse_args()

    r = calculate_metrics(args.predict_file, args.ground_truth_file)
    print("-" * 60)
    print("Evaluation Results")
    print("-" * 60)
    print(f"Total: {r['total_count']}")
    print(f"Overall accuracy: {r['overall_accuracy']:.4f}")
    print(f"Intent accuracy:  {r['intent_accuracy']:.4f}")
    print(f"Slot P/R/F1:      {r['slot_precision']:.4f} / {r['slot_recall']:.4f} / {r['slot_f1']:.4f}")
    print(f"Query MER:        {r['query_mer']:.4f} ({r['query_mer_errors']}/{r['query_mer_ref_len']})")
    print("-" * 60)


if __name__ == "__main__":
    main()
