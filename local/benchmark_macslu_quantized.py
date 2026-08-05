#!/usr/bin/env python3
"""Create a size/metric/latency comparison report from completed experiments."""
import argparse, json, re
from pathlib import Path

def size(path): return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
def metrics(path):
    text = Path(path).read_text() if Path(path).is_file() else ""
    def value(label):
        m = re.search(label + r"[^0-9]*([0-9.]+)", text, re.I); return float(m.group(1)) if m else None
    return {"accuracy": value("Overall accuracy"), "f1": value("Slot P/R/F1.*?/")}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--quantized_exp_dir", required=True); p.add_argument("--distillation_only_exp_dir", default=""); a=p.parse_args()
    q=Path(a.quantized_exp_dir); report=json.loads((q/"compression_report.json").read_text()); report["quantized_metrics"]=metrics(q/"test/metrics.txt")
    if a.distillation_only_exp_dir:
        d=Path(a.distillation_only_exp_dir); report["distillation_only_metrics"]=metrics(d/"test/metrics.txt"); report["distillation_only_checkpoint_bytes"]=size(d/"checkpoint-best")
        report["metric_absolute_difference"]={k: report["quantized_metrics"][k]-report["distillation_only_metrics"][k] if report["quantized_metrics"][k] is not None and report["distillation_only_metrics"][k] is not None else None for k in ("accuracy","f1")}
    else: report["comparison"]="not requested"
    report.setdefault("distillation_only_latency_seconds", None); report.setdefault("quantized_dense_latency_seconds", None); report["latency_note"]="Run inference stages under the same device and record wall time for latency comparison."
    (q/"comparison_report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
