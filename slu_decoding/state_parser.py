from dataclasses import dataclass

STATE_JSON_PREFIX = "STATE_JSON_PREFIX"
STATE_ASR_TEXT = "STATE_ASR_TEXT"
STATE_SEMANTICS = "STATE_SEMANTICS"
STATE_DOMAIN = "STATE_DOMAIN"
STATE_INTENT = "STATE_INTENT"
STATE_SLOTS_KEY = "STATE_SLOTS_KEY"
STATE_SLOTS_VALUE = "STATE_SLOTS_VALUE"
STATE_IMPLICIT_SLOTS_KEY = "STATE_IMPLICIT_SLOTS_KEY"
STATE_IMPLICIT_SLOTS_VALUE = "STATE_IMPLICIT_SLOTS_VALUE"


@dataclass
class DecodingState:
    state_name: str
    current_domain: str = ""
    current_intent: str = ""
    current_slot_key: str = ""
    inside_string: bool = False
    json_depth: int = 0


def _scan_jsonish(text: str):
    depth = 0
    inside = False
    escaped = False
    tokens = []
    buf = []

    def flush_buf():
        nonlocal buf
        if buf:
            tokens.append("".join(buf))
            buf = []

    for ch in text:
        if inside:
            if escaped:
                buf.append(ch)
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                flush_buf()
                inside = False
            else:
                buf.append(ch)
            continue

        if ch == '"':
            inside = True
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)

    return tokens, inside, depth


def parse_state(prefix_text: str) -> DecodingState:
    text = prefix_text or ""
    payload = text.split("<asr_text>", 1)[-1]
    tokens, inside, depth = _scan_jsonish(payload)

    current_domain = ""
    current_intent = ""
    current_slot_key = ""
    for idx, tok in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if tok == "domain":
            current_domain = nxt
        elif tok == "intent":
            current_intent = nxt

    lower = payload

    if '"semantics"' not in lower:
        return DecodingState(STATE_ASR_TEXT if '"asr_text"' in lower else STATE_JSON_PREFIX, inside_string=inside, json_depth=depth)

    if '"domain"' in lower:
        if lower.rstrip().endswith('"domain"') or ('"domain"' in lower and '"domain"' in lower[lower.rfind('"domain"'):] and ':' in lower[lower.rfind('"domain"'):] and lower[lower.rfind('"domain"'):].rstrip().endswith(':')):
            return DecodingState(STATE_DOMAIN, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)
        tail = lower[lower.rfind('"domain"'):]
        if ':' in tail:
            after = tail.split(':', 1)[1].strip()
            if after.startswith('"') and not after[1:].__contains__('"'):
                return DecodingState(STATE_DOMAIN, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)

    if '"intent"' in lower:
        if lower.rstrip().endswith('"intent"') or ('"intent"' in lower and '"intent"' in lower[lower.rfind('"intent"'):] and ':' in lower[lower.rfind('"intent"'):] and lower[lower.rfind('"intent"'):].rstrip().endswith(':')):
            return DecodingState(STATE_INTENT, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)
        tail = lower[lower.rfind('"intent"'):]
        if ':' in tail:
            after = tail.split(':', 1)[1].strip()
            if after.startswith('"') and not after[1:].__contains__('"'):
                return DecodingState(STATE_INTENT, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)

    if '"slots"' in lower:
        slots_tail = lower[lower.rfind('"slots"'):]
        if ':' in slots_tail:
            after_slots = slots_tail.split(':', 1)[1]
            # last opened quote not yet closed inside slots => key or value
            qpos = after_slots.rfind('"')
            if qpos >= 0:
                sub = after_slots[qpos + 1:]
                quote_count = after_slots[qpos:].count('"')
                if quote_count == 1:
                    prefix = after_slots[:qpos]
                    colon_after_key = ':' in prefix[prefix.rfind('{') + 1:] if '{' in prefix else ':' in prefix
                    if colon_after_key:
                        # likely value string
                        key_tokens = [t for t in tokens if t not in {"asr_text", "semantics", "domain", "intent", "slots", "implicit_slots"}]
                        current_slot_key = key_tokens[-1] if key_tokens else ""
                        return DecodingState(STATE_SLOTS_VALUE, current_domain=current_domain, current_intent=current_intent, current_slot_key=current_slot_key, inside_string=inside, json_depth=depth)
                    return DecodingState(STATE_SLOTS_KEY, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)

        if slots_tail.rstrip().endswith('{') or slots_tail.rstrip().endswith(','):
            return DecodingState(STATE_SLOTS_KEY, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)

    return DecodingState(STATE_SEMANTICS, current_domain=current_domain, current_intent=current_intent, inside_string=inside, json_depth=depth)
