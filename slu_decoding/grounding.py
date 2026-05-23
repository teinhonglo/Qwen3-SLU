import torch


def extract_candidate_spans(asr_text, min_n=1, max_n=4):
    s = (asr_text or "").strip()
    out = set()
    for n in range(min_n, max_n + 1):
        for i in range(max(0, len(s) - n + 1)):
            out.add(s[i : i + n])
    return sorted(out)


def build_copy_bias_map(tokenizer, asr_text):
    bias = {}
    for span in extract_candidate_spans(asr_text):
        ids = tokenizer.encode(span, add_special_tokens=False)
        for tid in ids:
            bias[tid] = max(bias.get(tid, 0.0), 1.0)
    return bias


def apply_copy_bias(logits, bias_map, strength=1.0):
    if not bias_map:
        return logits
    for tid, weight in bias_map.items():
        logits[..., tid] += strength * weight
    return logits