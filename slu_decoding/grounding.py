import torch


def build_copy_bias_map(tokenizer, asr_text):
    """Build token-level copy bias directly from ASR text tokenization.

    This avoids character n-gram expansion and uses tokenizer-native
    segmentation so grounding aligns with actual decoding token units.
    """
    s = (asr_text or "").strip()
    if not s:
        return {}

    ids = tokenizer.encode(s, add_special_tokens=False)
    return {int(tid): 1.0 for tid in ids}


def apply_copy_bias(logits, bias_map, strength=1.0):
    if not bias_map:
        return logits
    for tid, weight in bias_map.items():
        logits[..., tid] += strength * weight
    return logits