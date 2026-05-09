from dataclasses import dataclass
import re

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


def parse_state(prefix_text: str) -> DecodingState:
    text = prefix_text or ""
    payload = text.split("<asr_text>", 1)[-1]

    depth = 0
    inside = False
    for ch in payload:
        if ch == '"':
            inside = not inside
        if not inside and ch in "{[":
            depth += 1
        elif not inside and ch in "}]":
            depth = max(0, depth - 1)

    domains = re.findall(r'"domain"\s*:\s*"([^"]*)', payload)
    intents = re.findall(r'"intent"\s*:\s*"([^"]*)', payload)
    cur_domain = domains[-1] if domains else ""
    cur_intent = intents[-1] if intents else ""

    if '"asr_text"' in payload and payload.rstrip().endswith('"') and '"semantics"' not in payload:
        return DecodingState(STATE_ASR_TEXT, inside_string=inside, json_depth=depth)

    if re.search(r'"domain"\s*:\s*"?[^"]*$', payload):
        return DecodingState(
            STATE_DOMAIN,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    if re.search(r'"intent"\s*:\s*"?[^"]*$', payload):
        return DecodingState(
            STATE_INTENT,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    if '"implicit_slots"' in payload:
        if re.search(r'"implicit_slots"\s*:\s*\{[^}]*"[^"]*"\s*:\s*"?[^"]*$', payload):
            return DecodingState(
                STATE_IMPLICIT_SLOTS_VALUE,
                current_domain=cur_domain,
                current_intent=cur_intent,
                inside_string=inside,
                json_depth=depth,
            )
        return DecodingState(
            STATE_IMPLICIT_SLOTS_KEY,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    if '"slots"' in payload:
        if re.search(r'"slots"\s*:\s*\{[^}]*"[^"]*"\s*:\s*"?[^"]*$', payload):
            return DecodingState(
                STATE_SLOTS_VALUE,
                current_domain=cur_domain,
                current_intent=cur_intent,
                inside_string=inside,
                json_depth=depth,
            )
        return DecodingState(
            STATE_SLOTS_KEY,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    if '"semantics"' in payload:
        return DecodingState(
            STATE_SEMANTICS,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    return DecodingState(STATE_JSON_PREFIX, inside_string=inside, json_depth=depth)