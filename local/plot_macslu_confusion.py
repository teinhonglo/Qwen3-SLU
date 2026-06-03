import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


NONE_LABEL = "None"
EMPTY_LABEL = "Empty"
MISSING_MAPPING_FILENAME = "missing_label_mapping.txt"


@dataclass(frozen=True)
class LabelSchema:
    domains: List[str]
    domain_intents: List[Tuple[str, str]]
    intents_by_domain: Dict[str, List[str]]

    @property
    def valid_domains(self) -> set:
        return set(self.domains)

    @property
    def valid_domain_intents(self) -> set:
        return set(self.domain_intents)


@dataclass
class SemanticFrame:
    raw: Any
    domain: str
    intent: str
    domain_label: str
    intent_label: str
    reasons: List[str]
    malformed: bool = False


def clean_label(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def parse_label_schema(labels_file: str) -> LabelSchema:
    with open(labels_file, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"DOMAIN_INTENT_LIST\s*=\s*\"\"\"(.*?)\"\"\"", text, re.S)
    if match is None:
        raise ValueError(f"Cannot find DOMAIN_INTENT_LIST in {labels_file}")

    domains: List[str] = []
    domain_intents: List[Tuple[str, str]] = []
    intents_by_domain: Dict[str, List[str]] = {}
    current_domain: Optional[str] = None

    for line in match.group(1).splitlines():
        if not line.strip().startswith("- "):
            continue
        indent = len(line) - len(line.lstrip())
        label = clean_label(line.strip()[2:])
        if not label:
            continue
        if indent == 0:
            current_domain = label
            domains.append(label)
            intents_by_domain.setdefault(label, [])
        else:
            if current_domain is None:
                raise ValueError(f"Intent without a domain in {labels_file}: {line}")
            intents_by_domain[current_domain].append(label)
            domain_intents.append((current_domain, label))

    if not domains or not domain_intents:
        raise ValueError(f"No domain/intent labels parsed from {labels_file}")
    return LabelSchema(domains=domains, domain_intents=domain_intents, intents_by_domain=intents_by_domain)


def read_label_mapping(mapping_file: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid mapping line {line_no} in {mapping_file}. "
                    "Expected exactly two tab-separated columns."
                )
            zh, en = (part.strip() for part in parts)
            if not zh or not en:
                raise ValueError(f"Empty mapping value on line {line_no} in {mapping_file}")
            mapping[zh] = en
    return mapping


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def get_semantics(row: Dict[str, Any], key: str) -> List[Any]:
    value = row.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def intent_display_label(domain: str, intent: str) -> str:
    if intent in {NONE_LABEL, EMPTY_LABEL}:
        return intent
    if domain in {NONE_LABEL, EMPTY_LABEL, ""}:
        return intent
    return f"{domain} / {intent}"


def validate_gt_frame(frame: Any, schema: LabelSchema) -> SemanticFrame:
    if not isinstance(frame, dict):
        return SemanticFrame(frame, "", "", NONE_LABEL, NONE_LABEL, ["malformed_gt_frame"], malformed=True)

    domain = clean_label(frame.get("domain"))
    intent = clean_label(frame.get("intent"))
    domain_label = domain if domain in schema.valid_domains else (NONE_LABEL if domain else EMPTY_LABEL)
    if domain and intent and (domain, intent) in schema.valid_domain_intents:
        intent_label = intent_display_label(domain, intent)
    elif intent:
        intent_label = NONE_LABEL
    else:
        intent_label = EMPTY_LABEL

    reasons: List[str] = []
    if domain_label == NONE_LABEL:
        reasons.append("invalid_gt_domain")
    if intent_label == NONE_LABEL:
        reasons.append("invalid_gt_intent")
    return SemanticFrame(frame, domain, intent, domain_label, intent_label, reasons)


def validate_pred_frame(frame: Any, schema: LabelSchema) -> SemanticFrame:
    if not isinstance(frame, dict):
        return SemanticFrame(frame, "", "", NONE_LABEL, NONE_LABEL, ["malformed_frame"], malformed=True)

    domain = clean_label(frame.get("domain"))
    intent = clean_label(frame.get("intent"))
    reasons: List[str] = []

    if not domain:
        domain_label = EMPTY_LABEL
        if intent:
            intent_label = NONE_LABEL
            reasons.append("missing_domain")
        else:
            intent_label = EMPTY_LABEL
    elif domain not in schema.valid_domains:
        domain_label = NONE_LABEL
        intent_label = NONE_LABEL
        reasons.append("invalid_domain")
    else:
        domain_label = domain
        if not intent:
            intent_label = EMPTY_LABEL
            reasons.append("missing_intent")
        elif (domain, intent) in schema.valid_domain_intents:
            intent_label = intent_display_label(domain, intent)
        else:
            intent_label = NONE_LABEL
            valid_elsewhere = any((other_domain, intent) in schema.valid_domain_intents for other_domain in schema.domains)
            reasons.append("invalid_domain_intent_pair" if valid_elsewhere else "invalid_intent")

    return SemanticFrame(frame, domain, intent, domain_label, intent_label, reasons)


def frame_match_score(gt: SemanticFrame, pred: SemanticFrame) -> int:
    if gt.domain and pred.domain and gt.domain == pred.domain and gt.intent and pred.intent and gt.intent == pred.intent:
        return 100
    if gt.domain_label not in {NONE_LABEL, EMPTY_LABEL} and gt.domain_label == pred.domain_label:
        if gt.intent and pred.intent:
            return 80
        return 70
    if gt.intent and pred.intent and gt.intent == pred.intent:
        return 60
    if pred.domain_label == EMPTY_LABEL or pred.intent_label == EMPTY_LABEL:
        return 10
    if pred.domain_label == NONE_LABEL or pred.intent_label == NONE_LABEL:
        return 5
    return 1


def pair_frames(gt_frames: Sequence[SemanticFrame], pred_frames: Sequence[SemanticFrame]) -> List[Tuple[Optional[int], Optional[int]]]:
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    remaining_gt = set(range(len(gt_frames)))
    remaining_pred = set(range(len(pred_frames)))

    candidates: List[Tuple[int, int, int]] = []
    for gt_idx, gt in enumerate(gt_frames):
        for pred_idx, pred in enumerate(pred_frames):
            candidates.append((-frame_match_score(gt, pred), gt_idx, pred_idx))
    candidates.sort()

    for neg_score, gt_idx, pred_idx in candidates:
        if gt_idx in remaining_gt and pred_idx in remaining_pred:
            pairs.append((gt_idx, pred_idx))
            remaining_gt.remove(gt_idx)
            remaining_pred.remove(pred_idx)

    for gt_idx in sorted(remaining_gt):
        pairs.append((gt_idx, None))
    for pred_idx in sorted(remaining_pred):
        pairs.append((None, pred_idx))
    return pairs


def append_event(
    events: List[Dict[str, Any]],
    text_id: str,
    reason: str,
    gt: Optional[SemanticFrame],
    pred: Optional[SemanticFrame],
    gt_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    mapping: Dict[str, str],
) -> None:
    gt_domain = gt.domain if gt is not None else ""
    gt_intent = gt.intent if gt is not None else ""
    pred_domain_raw = pred.domain if pred is not None else ""
    pred_intent_raw = pred.intent if pred is not None else ""
    mapped_pred_domain = pred.domain_label if pred is not None else EMPTY_LABEL
    mapped_pred_intent = pred.intent_label if pred is not None else EMPTY_LABEL
    if " / " in mapped_pred_intent:
        _, mapped_pred_intent_plain = mapped_pred_intent.split(" / ", 1)
    else:
        mapped_pred_intent_plain = mapped_pred_intent

    events.append({
        "text_id": text_id,
        "reason": reason,
        "gt_domain": gt_domain,
        "gt_intent": gt_intent,
        "gt_domain_en": translate_label(gt_domain, mapping),
        "gt_intent_en": translate_label(gt_intent, mapping),
        "pred_domain_raw": pred_domain_raw,
        "pred_intent_raw": pred_intent_raw,
        "mapped_pred_domain": mapped_pred_domain,
        "mapped_pred_intent": mapped_pred_intent_plain,
        "mapped_pred_domain_en": translate_label(mapped_pred_domain, mapping),
        "mapped_pred_intent_en": translate_label(mapped_pred_intent_plain, mapping),
        "query": clean_label(gt_row.get("query", "")),
        "pred_query": clean_label(pred_row.get("pred_query", "")),
        "gt_frame": json.dumps(gt.raw if gt is not None else {}, ensure_ascii=False, sort_keys=True),
        "pred_frame": json.dumps(pred.raw if pred is not None else {}, ensure_ascii=False, sort_keys=True),
    })


def translate_label(label: str, mapping: Dict[str, str]) -> str:
    if label in {NONE_LABEL, EMPTY_LABEL, ""}:
        return label
    return mapping.get(label, label)


def translate_intent_display(label: str, mapping: Dict[str, str]) -> str:
    if label in {NONE_LABEL, EMPTY_LABEL, ""}:
        return label
    if " / " not in label:
        return translate_label(label, mapping)
    domain, intent = label.split(" / ", 1)
    return f"{translate_label(domain, mapping)} / {translate_label(intent, mapping)}"


def make_english_labels(labels: Sequence[str], mapping: Dict[str, str], intent: bool) -> List[str]:
    translated = [translate_intent_display(label, mapping) if intent else translate_label(label, mapping) for label in labels]
    counts: Counter = Counter()
    result: List[str] = []
    for label in translated:
        counts[label] += 1
        result.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    return result


def build_matrix(rows: Sequence[str], cols: Sequence[str], labels: Sequence[str]):
    import numpy as np
    import pandas as pd

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row_label, col_label in zip(rows, cols):
        if row_label not in index or col_label not in index:
            raise ValueError(f"Unknown confusion label: row={row_label}, col={col_label}")
        matrix[index[row_label], index[col_label]] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def normalize_matrix(count_df):
    import numpy as np
    import pandas as pd

    values = count_df.to_numpy(dtype=float)
    row_sum = values.sum(axis=1)
    normalized = np.divide(values, row_sum[:, np.newaxis], out=np.zeros_like(values), where=row_sum[:, np.newaxis] != 0)
    return pd.DataFrame(normalized, index=count_df.index, columns=count_df.columns)


def configure_fonts() -> None:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_heatmap(count_df, png_path: str, title: str, is_intent: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    norm_df = normalize_matrix(count_df)
    n_labels = len(count_df.index)
    if is_intent:
        width = max(18.0, min(48.0, n_labels * 0.48))
        height = max(16.0, min(48.0, n_labels * 0.42))
        tick_size = 9 if n_labels <= 60 else 8
        annot_size = 8 if n_labels <= 45 else 6
        rotation = 60
    else:
        width = max(10.0, n_labels * 0.85)
        height = max(8.0, n_labels * 0.72)
        tick_size = 11
        annot_size = 10
        rotation = 35

    plt.figure(figsize=(width, height))
    ax = sns.heatmap(
        data=norm_df,
        annot=count_df,
        fmt="g",
        cbar=False,
        cmap="Blues",
        linewidths=0.25,
        linecolor="white",
        annot_kws={"size": annot_size},
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predictions", fontsize=14)
    ax.set_ylabel("Annotations", fontsize=14)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right", fontsize=tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(png_path, dpi=250, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_count_and_normalized_csv(count_df, output_dir: str, stem: str) -> None:
    count_df.to_csv(os.path.join(output_dir, f"{stem}.csv"), encoding="utf-8-sig")
    normalize_matrix(count_df).to_csv(os.path.join(output_dir, f"{stem}_normalized.csv"), encoding="utf-8-sig")


def save_hallucination_report(
    output_dir: str,
    events: Sequence[Dict[str, Any]],
    pred_file: str,
    gt_file: str,
    labels_file: str,
    mapping_file: str,
    total_records: int,
) -> None:
    out_path = os.path.join(output_dir, "hallucination.txt")
    reason_counts = Counter(event["reason"] for event in events)
    fieldnames = [
        "text_id",
        "reason",
        "gt_domain",
        "gt_intent",
        "gt_domain_en",
        "gt_intent_en",
        "pred_domain_raw",
        "pred_intent_raw",
        "mapped_pred_domain",
        "mapped_pred_intent",
        "mapped_pred_domain_en",
        "mapped_pred_intent_en",
        "query",
        "pred_query",
        "gt_frame",
        "pred_frame",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# Hallucination / Missing Prediction Report\n")
        f.write(f"# pred_file: {pred_file}\n")
        f.write(f"# gt_file: {gt_file}\n")
        f.write(f"# labels_file: {labels_file}\n")
        f.write(f"# label_mapping_file: {mapping_file}\n")
        f.write(f"# total_records: {total_records}\n")
        f.write(f"# total_events: {len(events)}\n")
        f.write("# reason_counts:\n")
        for reason, count in sorted(reason_counts.items()):
            f.write(f"#   {reason}: {count}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def save_missing_mapping_report(output_dir: str, schema: LabelSchema, mapping: Dict[str, str]) -> None:
    labels = list(dict.fromkeys(schema.domains + [intent for _, intent in schema.domain_intents]))
    missing = [label for label in labels if label not in mapping]
    out_path = os.path.join(output_dir, MISSING_MAPPING_FILENAME)
    if not missing:
        if os.path.exists(out_path):
            os.remove(out_path)
        return
    with open(out_path, "w", encoding="utf-8") as f:
        for label in missing:
            f.write(label + "\n")
    print(f"[WARNING] Missing English label mappings written to {out_path}")


def collect_confusion_data(
    pred_rows: Sequence[Dict[str, Any]],
    gt_rows: Sequence[Dict[str, Any]],
    schema: LabelSchema,
    mapping: Dict[str, str],
) -> Tuple[List[str], List[str], List[str], List[str], List[Dict[str, Any]]]:
    domain_gt: List[str] = []
    domain_pred: List[str] = []
    intent_gt: List[str] = []
    intent_pred: List[str] = []
    events: List[Dict[str, Any]] = []

    if len(pred_rows) != len(gt_rows):
        raise ValueError(f"Line count mismatch: pred={len(pred_rows)} gt={len(gt_rows)}")

    for row_idx, (pred_row, gt_row) in enumerate(zip(pred_rows, gt_rows), start=1):
        text_id = clean_label(gt_row.get("text_id") or pred_row.get("text_id") or f"line_{row_idx}")
        gt_frames = [validate_gt_frame(frame, schema) for frame in get_semantics(gt_row, "semantics")]
        pred_frames = [validate_pred_frame(frame, schema) for frame in get_semantics(pred_row, "pred_semantics")]
        pairs = pair_frames(gt_frames, pred_frames)

        for gt_idx, pred_idx in pairs:
            gt = gt_frames[gt_idx] if gt_idx is not None else None
            pred = pred_frames[pred_idx] if pred_idx is not None else None

            gt_domain_label = gt.domain_label if gt is not None else EMPTY_LABEL
            gt_intent_label = gt.intent_label if gt is not None else EMPTY_LABEL
            pred_domain_label = pred.domain_label if pred is not None else EMPTY_LABEL
            pred_intent_label = pred.intent_label if pred is not None else EMPTY_LABEL

            domain_gt.append(gt_domain_label)
            domain_pred.append(pred_domain_label)
            intent_gt.append(gt_intent_label)
            intent_pred.append(pred_intent_label)

            if gt is None and pred is not None:
                append_event(events, text_id, "extra_frame", gt, pred, gt_row, pred_row, mapping)

            if gt is not None and pred is None:
                append_event(events, text_id, "missing_frame", gt, pred, gt_row, pred_row, mapping)
                continue

            if pred is None:
                continue

            for reason in pred.reasons:
                if reason == "missing_domain" and (gt is None or not gt.domain):
                    continue
                if reason == "missing_intent" and (gt is None or not gt.intent):
                    continue
                append_event(events, text_id, reason, gt, pred, gt_row, pred_row, mapping)

            if gt is not None and gt.domain and pred.domain_label == EMPTY_LABEL and "missing_domain" not in pred.reasons:
                append_event(events, text_id, "missing_domain", gt, pred, gt_row, pred_row, mapping)
            if gt is not None and gt.intent and pred.intent_label == EMPTY_LABEL and "missing_intent" not in pred.reasons:
                append_event(events, text_id, "missing_intent", gt, pred, gt_row, pred_row, mapping)

    return domain_gt, domain_pred, intent_gt, intent_pred, events


def relabel_dataframe(df, labels: Sequence[str]):
    relabeled = df.copy()
    relabeled.index = labels
    relabeled.columns = labels
    return relabeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAC-SLU domain and intent confusion matrices.")
    parser.add_argument("--pred_file", required=True, help="Prediction JSONL with pred_semantics.")
    parser.add_argument("--gt_file", required=True, help="Ground-truth JSONL with semantics.")
    parser.add_argument("--labels_file", required=True, help="MAC-SLU labels.txt schema file.")
    parser.add_argument("--label_mapping_file", required=True, help="Tab-separated Chinese-to-English label mapping file.")
    parser.add_argument("--output_dir", required=True, help="Directory for PNG, CSV, and hallucination outputs.")
    parser.add_argument("--skip_plots", action="store_true", help="Write CSV/report outputs without rendering PNG files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    schema = parse_label_schema(args.labels_file)
    mapping = read_label_mapping(args.label_mapping_file)
    save_missing_mapping_report(args.output_dir, schema, mapping)

    pred_rows = load_jsonl(args.pred_file)
    gt_rows = load_jsonl(args.gt_file)
    domain_gt, domain_pred, intent_gt, intent_pred, events = collect_confusion_data(pred_rows, gt_rows, schema, mapping)

    domain_labels = schema.domains + [NONE_LABEL, EMPTY_LABEL]
    intent_labels = [intent_display_label(domain, intent) for domain, intent in schema.domain_intents] + [NONE_LABEL, EMPTY_LABEL]

    domain_count_df = build_matrix(domain_gt, domain_pred, domain_labels)
    intent_count_df = build_matrix(intent_gt, intent_pred, intent_labels)

    save_count_and_normalized_csv(domain_count_df, args.output_dir, "domain_confusion_matrix")
    save_count_and_normalized_csv(intent_count_df, args.output_dir, "intent_confusion_matrix")
    if not args.skip_plots:
        configure_fonts()
        plot_heatmap(domain_count_df, os.path.join(args.output_dir, "domain_confusion_matrix.png"), "Domain Confusion Matrix", False)
        plot_heatmap(intent_count_df, os.path.join(args.output_dir, "intent_confusion_matrix.png"), "Intent Confusion Matrix", True)

    domain_labels_en = make_english_labels(domain_labels, mapping, intent=False)
    intent_labels_en = make_english_labels(intent_labels, mapping, intent=True)
    domain_count_en_df = relabel_dataframe(domain_count_df, domain_labels_en)
    intent_count_en_df = relabel_dataframe(intent_count_df, intent_labels_en)

    save_count_and_normalized_csv(domain_count_en_df, args.output_dir, "domain_confusion_matrix_en")
    save_count_and_normalized_csv(intent_count_en_df, args.output_dir, "intent_confusion_matrix_en")
    if not args.skip_plots:
        plot_heatmap(domain_count_en_df, os.path.join(args.output_dir, "domain_confusion_matrix_en.png"), "Domain Confusion Matrix", False)
        plot_heatmap(intent_count_en_df, os.path.join(args.output_dir, "intent_confusion_matrix_en.png"), "Intent Confusion Matrix", True)

    save_hallucination_report(
        args.output_dir,
        events,
        args.pred_file,
        args.gt_file,
        args.labels_file,
        args.label_mapping_file,
        len(gt_rows),
    )

    if args.skip_plots:
        print("Skipped PNG rendering because --skip_plots was set")
    else:
        print(f"Saved domain confusion matrix to {os.path.join(args.output_dir, 'domain_confusion_matrix.png')}")
        print(f"Saved intent confusion matrix to {os.path.join(args.output_dir, 'intent_confusion_matrix.png')}")
    print(f"Saved hallucination report to {os.path.join(args.output_dir, 'hallucination.txt')}")


if __name__ == "__main__":
    main()
