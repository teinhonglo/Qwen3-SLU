#!/usr/bin/env python3
"""Plot MAC-SLU train/test instances and prototypes with t-SNE."""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

SEP = "|||"


# ============================================================
# 0. Linux 中文字體設定
# ============================================================
def setup_chinese_font():
    """
    Setup Chinese font for Matplotlib on Linux.

    Recommended installation:
        sudo apt-get update
        sudo apt-get install -y fonts-noto-cjk

    If Chinese characters still cannot be displayed:
        rm -rf ~/.cache/matplotlib
    """
    possible_fonts = [
        # Ubuntu / Debian common Noto CJK paths
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",

        # Other possible Noto CJK names
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",

        # AR PL fonts, sometimes available on Linux
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            # Critical: explicitly register the font file.
            fm.fontManager.addfont(font_path)

            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()

            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False

            # Better PDF/PS font embedding.
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42

            print(f"[info] 使用中文字體: {font_name} ({font_path})")
            return font_prop

    raise FileNotFoundError(
        "找不到 Linux 中文字體。請先安裝：\n"
        "    sudo apt-get update\n"
        "    sudo apt-get install -y fonts-noto-cjk\n"
        "若安裝後仍失敗，請清除 Matplotlib cache：\n"
        "    rm -rf ~/.cache/matplotlib"
    )


FONT_PROP = setup_chinese_font()


def apply_font_to_legend(legend):
    """Apply Chinese font to all legend texts."""
    if legend is None:
        return

    legend.get_title().set_fontproperties(FONT_PROP)

    for text in legend.get_texts():
        text.set_fontproperties(FONT_PROP)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"JSON object expected: {path}")

    return obj


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    if not path:
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_id}: {exc}") from exc

    return rows


def safe_name(text: str) -> str:
    text = str(text or "unknown")
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    return text[:120] or "unknown"


def sample_rows(
    rows: Iterable[Dict[str, Any]],
    max_per_label: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_per_label <= 0:
        return list(rows)

    groups = defaultdict(list)

    for row in rows:
        groups[row.get("label", "")].append(row)

    rng = random.Random(seed)
    sampled = []

    for label in sorted(groups):
        vals = list(groups[label])
        rng.shuffle(vals)
        sampled.extend(vals[:max_per_label])

    return sampled


def prototype_rows(
    prototype_json: Dict[str, Any],
    kind: str,
    domain: str = "",
) -> List[Dict[str, Any]]:
    rows = []

    for key, item in (prototype_json.get(kind, {}) or {}).items():
        if not isinstance(item, dict):
            continue

        meta = item.get("meta", {}) or {}

        if domain and meta.get("domain", "") != domain:
            continue

        vec = item.get("vector", []) or []

        if not vec:
            continue

        rows.append(
            {
                "split": "prototype",
                "kind": kind,
                "key": key,
                "label": meta.get("label", key.split(SEP)[-1]),
                "domain": meta.get("domain", key if kind == "domain" else ""),
                "intent": meta.get("intent", ""),
                "vector": vec,
                "count": item.get("count", 0),
            }
        )

    return rows


def effective_perplexity(n_points: int, requested: float) -> float:
    if n_points <= 3:
        return 1.0

    return float(
        max(
            2,
            min(
                float(requested),
                max(2, (n_points - 1) // 3),
                n_points - 1,
            ),
        )
    )


def run_tsne(
    rows: List[Dict[str, Any]],
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    vectors = np.asarray([row["vector"] for row in rows], dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] < 3:
        raise ValueError("Need at least 3 vectors for t-SNE")

    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity(vectors.shape[0], perplexity),
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )

    return tsne.fit_transform(vectors)


def color_map(labels: List[str], plt_module):
    unique = sorted(set(labels))
    cmap_name = "tab20" if len(unique) <= 20 else "hsv"
    cmap = plt_module.get_cmap(cmap_name, max(len(unique), 1))

    return {label: cmap(i) for i, label in enumerate(unique)}


def plot_rows(
    rows: List[Dict[str, Any]],
    out_base: str,
    title: str,
    perplexity: float,
    random_state: int,
    annotate_prototypes: bool,
) -> Dict[str, Any]:
    if len(rows) < 3:
        return {
            "path": out_base,
            "status": "skipped",
            "reason": "fewer_than_3_points",
            "n_points": len(rows),
        }

    coords = run_tsne(
        rows,
        perplexity=perplexity,
        random_state=random_state,
    )

    labels = [str(row.get("label", "")) for row in rows]
    colors = color_map(labels, plt)

    markers = {
        "train": "o",
        "test": "^",
        "prototype": "*",
    }

    sizes = {
        "train": 26,
        "test": 42,
        "prototype": 260,
    }

    alphas = {
        "train": 0.35,
        "test": 0.62,
        "prototype": 1.0,
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for split in ("train", "test", "prototype"):
        idxs = [
            i
            for i, row in enumerate(rows)
            if row.get("split") == split
        ]

        if not idxs:
            continue

        for label in sorted({labels[i] for i in idxs}):
            label_idxs = [
                i
                for i in idxs
                if labels[i] == label
            ]

            edgecolors = "black" if split == "prototype" else "none"
            linewidths = 1.1 if split == "prototype" else 0.0

            ax.scatter(
                coords[label_idxs, 0],
                coords[label_idxs, 1],
                c=[colors[label]],
                marker=markers[split],
                s=sizes[split],
                alpha=alphas[split],
                edgecolors=edgecolors,
                linewidths=linewidths,
            )

            if split == "prototype" and annotate_prototypes:
                for i in label_idxs:
                    ax.annotate(
                        label,
                        (coords[i, 0], coords[i, 1]),
                        fontsize=9,
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontproperties=FONT_PROP,
                    )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=colors[label],
            markersize=8,
            linestyle="None",
        )
        for label in sorted(colors)
    ]

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="train instance",
            linestyle="None",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            label="test instance",
            linestyle="None",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="gray",
            label="prototype",
            linestyle="None",
            markersize=13,
        ),
    ]

    legend1 = ax.legend(
        handles=marker_handles,
        loc="upper right",
        title="Point type",
    )
    apply_font_to_legend(legend1)
    ax.add_artist(legend1)

    if len(class_handles) <= 25:
        legend2 = ax.legend(
            handles=class_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Class",
        )
        apply_font_to_legend(legend2)

    ax.set_title(title, fontproperties=FONT_PROP)
    ax.set_xlabel("t-SNE 1", fontproperties=FONT_PROP)
    ax.set_ylabel("t-SNE 2", fontproperties=FONT_PROP)

    ax.grid(alpha=0.2)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    png = f"{out_base}.png"
    pdf = f"{out_base}.pdf"

    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)

    return {
        "path": out_base,
        "status": "saved",
        "png": png,
        "pdf": pdf,
        "n_points": len(rows),
        "n_classes": len(colors),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--prototype_json", required=True)
    p.add_argument("--train_examples_jsonl", required=True)
    p.add_argument("--test_examples_jsonl", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_train_examples_per_label", type=int, default=200)
    p.add_argument("--max_test_examples_per_label", type=int, default=200)
    p.add_argument("--random_state", type=int, default=66)

    p.add_argument(
        "--annotate_prototypes",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no_annotate_prototypes",
        dest="annotate_prototypes",
        action="store_false",
    )

    return p.parse_args()


def main():
    args = parse_args()

    prototype_json = load_json(args.prototype_json)
    train_rows = load_jsonl(args.train_examples_jsonl)
    test_rows = load_jsonl(args.test_examples_jsonl)

    manifest = {
        "domain": None,
        "intent_by_domain": [],
    }

    domain_rows = (
        sample_rows(
            [r for r in train_rows if r.get("kind") == "domain"],
            args.max_train_examples_per_label,
            args.random_state,
        )
        + sample_rows(
            [r for r in test_rows if r.get("kind") == "domain"],
            args.max_test_examples_per_label,
            args.random_state,
        )
        + prototype_rows(prototype_json, "domain")
    )

    manifest["domain"] = plot_rows(
        domain_rows,
        os.path.join(args.output_dir, "domain_tsne"),
        "MAC-SLU Domain Prototype t-SNE (train/test/prototype)",
        args.perplexity,
        args.random_state,
        args.annotate_prototypes,
    )

    domains = sorted(
        set(
            r.get("domain", "")
            for r in train_rows + test_rows
            if r.get("kind") == "intent" and r.get("domain", "")
        )
        | set(
            r.get("domain", "")
            for r in prototype_rows(prototype_json, "intent")
            if r.get("domain", "")
        )
    )

    intent_dir = os.path.join(args.output_dir, "intent_tsne")

    for domain in domains:
        rows = (
            sample_rows(
                [
                    r
                    for r in train_rows
                    if r.get("kind") == "intent"
                    and r.get("domain") == domain
                ],
                args.max_train_examples_per_label,
                args.random_state,
            )
            + sample_rows(
                [
                    r
                    for r in test_rows
                    if r.get("kind") == "intent"
                    and r.get("domain") == domain
                ],
                args.max_test_examples_per_label,
                args.random_state,
            )
            + prototype_rows(
                prototype_json,
                "intent",
                domain=domain,
            )
        )

        manifest["intent_by_domain"].append(
            plot_rows(
                rows,
                os.path.join(
                    intent_dir,
                    f"{safe_name(domain)}_intent_tsne",
                ),
                f"MAC-SLU Intent Prototype t-SNE - Domain: {domain}",
                args.perplexity,
                args.random_state,
                args.annotate_prototypes,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(
        args.output_dir,
        "prototype_tsne_manifest.json",
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[info] saved t-SNE manifest: {manifest_path}")


if __name__ == "__main__":
    main()