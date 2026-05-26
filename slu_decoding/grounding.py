import torch
import re


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


def trim_asr_text_left_of_decoded_value(asr_text, decoded_prefix):
    """Ignore left-side ASR text up to and including an already decoded slot value.

    When slot value content has already appeared in ASR text, we keep only the
    right-side suffix so copy bias focuses on not-yet-consumed text.
    """
    s = asr_text or ""
    if not s or not decoded_prefix:
        return s

    # Capture the currently decoded slot value fragment in slots object.
    # Supports both complete and still-open strings.
    m = re.search(r'"slots"\s*:\s*\{.*?"[^"]*"\s*:\s*"(?P<val>(?:\\.|[^"\\])*)$', decoded_prefix)
    if not m:
        m = re.search(r'"slots"\s*:\s*\{.*?"[^"]*"\s*:\s*"(?P<val>(?:\\.|[^"\\])*)"', decoded_prefix)
    if not m:
        return s

    val = m.group("val").replace('\\"', '"').strip()
    if not val:
        return s

    idx = s.find(val)
    if idx < 0:
        return s
    return s[idx + len(val):]


def apply_copy_bias(logits, bias_map, strength=1.0):
    if not bias_map:
        return logits
    for tid, weight in bias_map.items():
        logits[..., tid] += strength * weight
    return logits
