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
OTHER_DOMAIN_LABEL = "Other domain"


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


def safe_filename(text: str) -> str:
    text = str(text or "unknown")
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    return text[:120] or "unknown"


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


def configure_fonts():
    """
    Configure Matplotlib fonts for Chinese labels on Linux.

    Recommended installation on Ubuntu/Debian:
        sudo apt-get update
        sudo apt-get install -y fonts-noto-cjk

    If the font was installed after Matplotlib was first used, clear cache:
        rm -rf ~/.cache/matplotlib
    """
    import os
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    possible_font_paths = [
        # Ubuntu / Debian Noto CJK package
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",

        # Sometimes installed with region-specific names
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",

        # AR PL fonts on some Linux systems
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]

    for font_path in possible_font_paths:
        if os.path.exists(font_path):
            # Critical: register the exact font file, not only the font name.
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()

            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False

            # Better font embedding for vector outputs if PDF/PS are added later.
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42

            print(f"[info] 使用中文字體: {font_name} ({font_path})")
            return font_prop

    raise FileNotFoundError(
        "找不到 Linux 中文字體。請先安裝：\n"
        "    sudo apt-get update\n"
        "    sudo apt-get install -y fonts-noto-cjk\n"
        "若安裝後仍無法顯示中文，請清除 Matplotlib cache：\n"
        "    rm -rf ~/.cache/matplotlib"
    )


def apply_font_to_heatmap_axis(ax, font_prop, tick_size: int) -> None:
    """Apply Chinese font to labels, tick labels, and heatmap annotations."""
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontproperties(font_prop)
        tick_label.set_fontsize(tick_size)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontproperties(font_prop)
        tick_label.set_fontsize(tick_size)

    # Seaborn heatmap annotations are stored as ax.texts.
    for text in ax.texts:
        text.set_fontproperties(font_prop)


def choose_heatmap_layout(n_rows: int, n_cols: int, is_intent: bool) -> Tuple[float, float, int, int, int, int]:
    """
    Return width, height, tick_size, annotation_size, rotation, dpi.

    The key idea is to scale by the number of rows/columns instead of
    compressing all labels into a fixed-size canvas. For very large matrices,
    the PNG dpi is reduced to keep the pixel dimensions reasonable, while a PDF
    is always saved for zoomable/vector inspection.
    """
    n_max = max(n_rows, n_cols)

    if is_intent:
        width = max(24.0, min(140.0, n_cols * 0.78))
        height = max(20.0, min(140.0, n_rows * 0.62))
        rotation = 70
        if n_max <= 45:
            tick_size, annot_size, dpi = 10, 8, 250
        elif n_max <= 90:
            tick_size, annot_size, dpi = 9, 7, 220
        elif n_max <= 140:
            tick_size, annot_size, dpi = 8, 6, 180
        else:
            tick_size, annot_size, dpi = 7, 5, 140
    else:
        width = max(10.0, min(80.0, n_cols * 0.95))
        height = max(8.0, min(80.0, n_rows * 0.78))
        rotation = 40
        tick_size, annot_size, dpi = 11, 10, 250

    return width, height, tick_size, annot_size, rotation, dpi


def plot_heatmap(
    count_df,
    png_path: str,
    title: str,
    is_intent: bool,
    annotate: Optional[bool] = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    font_prop = configure_fonts()
    norm_df = normalize_matrix(count_df)
    n_rows, n_cols = count_df.shape
    n_max = max(n_rows, n_cols)

    width, height, tick_size, annot_size, rotation, dpi = choose_heatmap_layout(
        n_rows=n_rows,
        n_cols=n_cols,
        is_intent=is_intent,
    )

    # Full matrices with too many labels become visually noisy if every cell is annotated.
    # The corresponding CSV files still contain the exact counts. Smaller per-domain plots
    # remain annotated by default.
    if annotate is None:
        annotate = n_max <= 90

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        data=norm_df,
        annot=count_df if annotate else False,
        fmt="g",
        cbar=False,
        cmap="Blues",
        linewidths=0.25,
        linecolor="white",
        annot_kws={"size": annot_size, "fontproperties": font_prop},
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )

    ax.set_xlabel("Predictions", fontsize=max(14, tick_size + 4), fontproperties=font_prop)
    ax.set_ylabel("Annotations", fontsize=max(14, tick_size + 4), fontproperties=font_prop)
    ax.set_title(title, fontsize=max(16, tick_size + 6), pad=14, fontproperties=font_prop)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_size,
        fontproperties=font_prop,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=tick_size,
        fontproperties=font_prop,
    )

    apply_font_to_heatmap_axis(ax, font_prop, tick_size)

    # Avoid tight_layout squeezing the heatmap itself when labels are long.
    # bbox_inches="tight" below still keeps all labels in the output image.
    fig.subplots_adjust(left=0.22, bottom=0.28, right=0.98, top=0.94)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def drop_empty_rows_and_columns(count_df, keep_rows: Optional[Sequence[str]] = None, keep_cols: Optional[Sequence[str]] = None):
    """Remove all-zero rows/columns, but keep requested labels even if they are all zero."""
    keep_rows = set(keep_rows or [])
    keep_cols = set(keep_cols or [])

    row_mask = (count_df.sum(axis=1) > 0) | count_df.index.to_series().isin(keep_rows).to_numpy()
    col_mask = (count_df.sum(axis=0) > 0) | count_df.columns.to_series().isin(keep_cols).to_numpy()

    if row_mask.any():
        count_df = count_df.loc[row_mask]
    if col_mask.any():
        count_df = count_df.loc[:, col_mask]
    return count_df


def build_domain_intent_matrix(full_count_df, domain: str, intents: Sequence[str], drop_empty: bool = True):
    import pandas as pd

    domain_labels = [intent_display_label(domain, intent) for intent in intents]
    explicit_lookup_cols = domain_labels + [NONE_LABEL, EMPTY_LABEL]
    # Keep Empty as the final displayed class, and always keep Other domain visible.
    output_cols = domain_labels + [NONE_LABEL, OTHER_DOMAIN_LABEL, EMPTY_LABEL]

    rows = []
    for row_label in domain_labels:
        if row_label not in full_count_df.index:
            continue

        row_values = {}
        for col_label in domain_labels + [NONE_LABEL, EMPTY_LABEL]:
            row_values[col_label] = int(full_count_df.loc[row_label, col_label]) if col_label in full_count_df.columns else 0

        other_sum = 0
        for col_label in full_count_df.columns:
            if col_label not in explicit_lookup_cols:
                other_sum += int(full_count_df.loc[row_label, col_label])
        row_values[OTHER_DOMAIN_LABEL] = other_sum
        rows.append((row_label, row_values))

    if not rows:
        return pd.DataFrame(columns=output_cols)

    out_df = pd.DataFrame([values for _, values in rows], index=[label for label, _ in rows], columns=output_cols)
    if drop_empty:
        out_df = drop_empty_rows_and_columns(
            out_df,
            keep_cols=[NONE_LABEL, OTHER_DOMAIN_LABEL, EMPTY_LABEL],
        )
    return out_df


def relabel_rectangular_dataframe(df, row_labels: Sequence[str], col_labels: Sequence[str]):
    relabeled = df.copy()
    relabeled.index = row_labels
    relabeled.columns = col_labels
    return relabeled


def relabel_domain_intent_dataframe(df, mapping: Dict[str, str]):
    row_labels = [translate_intent_display(label, mapping) for label in df.index]
    col_labels = [
        OTHER_DOMAIN_LABEL if label == OTHER_DOMAIN_LABEL else translate_intent_display(label, mapping)
        for label in df.columns
    ]
    return relabel_rectangular_dataframe(df, row_labels, col_labels)


def save_domain_intent_plots(
    full_count_df,
    schema: LabelSchema,
    mapping: Dict[str, str],
    output_dir: str,
    english: bool,
    drop_empty: bool,
) -> None:
    subdir = "intent_by_domain_en" if english else "intent_by_domain"
    plot_dir = os.path.join(output_dir, subdir)
    os.makedirs(plot_dir, exist_ok=True)

    for domain in schema.domains:
        domain_df = build_domain_intent_matrix(
            full_count_df,
            domain=domain,
            intents=schema.intents_by_domain.get(domain, []),
            drop_empty=drop_empty,
        )
        if domain_df.empty or int(domain_df.to_numpy().sum()) == 0:
            continue

        display_domain = translate_label(domain, mapping) if english else domain
        if english:
            domain_df = relabel_domain_intent_dataframe(domain_df, mapping)

        stem = safe_filename(display_domain)
        save_count_and_normalized_csv(domain_df, plot_dir, stem)
        plot_heatmap(
            domain_df,
            os.path.join(plot_dir, f"{stem}.png"),
            f"Intent Confusion Matrix - {display_domain}",
            is_intent=True,
            annotate=True,
        )


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
    parser.add_argument(
        "--no_intent_by_domain_plots",
        action="store_true",
        help="Do not render additional per-domain intent confusion matrices.",
    )
    parser.add_argument(
        "--keep_empty_domain_intents",
        action="store_true",
        help="Keep all intent labels in per-domain plots, including labels with zero counts.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    schema = parse_label_schema(args.labels_file)
    mapping = read_label_mapping(args.label_mapping_file)
    save_missing_mapping_report(args.output_dir, schema, mapping)

    pred_rows = load_jsonl(args.pred_file)
    gt_rows = load_jsonl(args.gt_file)
    domain_gt, domain_pred, domain_intent_gt, domain_intent_pred, events = collect_confusion_data(
        pred_rows, gt_rows, schema, mapping
    )

    domain_labels = schema.domains + [NONE_LABEL, EMPTY_LABEL]
    domain_intent_labels = [
        intent_display_label(domain, intent) for domain, intent in schema.domain_intents
    ] + [NONE_LABEL, EMPTY_LABEL]

    domain_count_df = build_matrix(domain_gt, domain_pred, domain_labels)
    domain_intent_count_df = build_matrix(domain_intent_gt, domain_intent_pred, domain_intent_labels)

    save_count_and_normalized_csv(domain_count_df, args.output_dir, "domain_confusion_matrix")
    # Keep the historical intent_* output names for compatibility.  These labels are
    # already domain-scoped ("domain / intent"), so also write explicit
    # domain_intent_* aliases to make the joint confusion matrix easy to find.
    save_count_and_normalized_csv(domain_intent_count_df, args.output_dir, "intent_confusion_matrix")
    save_count_and_normalized_csv(domain_intent_count_df, args.output_dir, "domain_intent_confusion_matrix")
    if not args.skip_plots:
        configure_fonts()
        plot_heatmap(domain_count_df, os.path.join(args.output_dir, "domain_confusion_matrix.png"), "Domain Confusion Matrix", False)
        plot_heatmap(
            domain_intent_count_df,
            os.path.join(args.output_dir, "intent_confusion_matrix.png"),
            "Intent Confusion Matrix",
            True,
            annotate=None,
        )
        plot_heatmap(
            domain_intent_count_df,
            os.path.join(args.output_dir, "domain_intent_confusion_matrix.png"),
            "Domain-Intent Confusion Matrix",
            True,
            annotate=None,
        )
        if not args.no_intent_by_domain_plots:
            save_domain_intent_plots(
                domain_intent_count_df,
                schema=schema,
                mapping=mapping,
                output_dir=args.output_dir,
                english=False,
                drop_empty=not args.keep_empty_domain_intents,
            )

    domain_labels_en = make_english_labels(domain_labels, mapping, intent=False)
    domain_intent_labels_en = make_english_labels(domain_intent_labels, mapping, intent=True)
    domain_count_en_df = relabel_dataframe(domain_count_df, domain_labels_en)
    domain_intent_count_en_df = relabel_dataframe(domain_intent_count_df, domain_intent_labels_en)

    save_count_and_normalized_csv(domain_count_en_df, args.output_dir, "domain_confusion_matrix_en")
    save_count_and_normalized_csv(domain_intent_count_en_df, args.output_dir, "intent_confusion_matrix_en")
    save_count_and_normalized_csv(domain_intent_count_en_df, args.output_dir, "domain_intent_confusion_matrix_en")
    if not args.skip_plots:
        plot_heatmap(domain_count_en_df, os.path.join(args.output_dir, "domain_confusion_matrix_en.png"), "Domain Confusion Matrix", False)
        plot_heatmap(
            domain_intent_count_en_df,
            os.path.join(args.output_dir, "intent_confusion_matrix_en.png"),
            "Intent Confusion Matrix",
            True,
            annotate=None,
        )
        plot_heatmap(
            domain_intent_count_en_df,
            os.path.join(args.output_dir, "domain_intent_confusion_matrix_en.png"),
            "Domain-Intent Confusion Matrix",
            True,
            annotate=None,
        )
        if not args.no_intent_by_domain_plots:
            save_domain_intent_plots(
                domain_intent_count_df,
                schema=schema,
                mapping=mapping,
                output_dir=args.output_dir,
                english=True,
                drop_empty=not args.keep_empty_domain_intents,
            )

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
        print(
            f"Saved domain-intent confusion matrix to "
            f"{os.path.join(args.output_dir, 'domain_intent_confusion_matrix.png')}"
        )
        if not args.no_intent_by_domain_plots:
            print(f"Saved per-domain intent matrices to {os.path.join(args.output_dir, 'intent_by_domain')}")
    print(f"Saved hallucination report to {os.path.join(args.output_dir, 'hallucination.txt')}")


if __name__ == "__main__":
    main()