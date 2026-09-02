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
    active_label_prefix: str = ""
    active_label_quote: str = ""
    active_label: bool = False
    inside_string: bool = False
    json_depth: int = 0

Q = r'(?:\\"|")'

def key_re(name: str) -> str:
    return rf'{Q}{re.escape(name)}{Q}'


def _active_value(text: str, key_name: str):
    """Return the unfinished string value for ``key_name`` at the prefix end."""
    pattern = (
        rf'{key_re(key_name)}\s*:\s*'
        rf'(?P<quote>\\"|")(?P<value>(?:\\(?!")|[^"\\])*)$'
    )
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group("value"), match.group("quote")


def _latest_open_object(text: str):
    """Return the latest still-open ``slots``/``implicit_slots`` object."""
    matches = []
    for key_name in ("slots", "implicit_slots"):
        for match in re.finditer(rf'{key_re(key_name)}\s*:\s*\{{', text):
            matches.append((match.start(), match.end(), key_name))
    if not matches:
        return None

    _, end, key_name = max(matches, key=lambda item: item[0])
    tail = text[end:]
    # MAC-SLU slot objects do not contain nested objects.  A closing brace after
    # the latest object opener therefore means this field has already ended.
    if "}" in tail:
        return None
    return key_name, tail


def _active_object_key(object_tail: str):
    """Return an unfinished slot-key string at the end of an open object."""
    match = re.search(
        r'(?:^|,)\s*(?P<quote>\\"|")'
        r'(?P<value>(?:\\(?!")|[^"\\])*)$',
        object_tail,
    )
    if match is None:
        return None
    return match.group("value"), match.group("quote")


def _active_object_value(object_tail: str):
    """Return the current key when its unfinished value is at the prefix end."""
    match = re.search(
        rf'{Q}(?P<key>(?:\\(?!")|[^"\\])*){Q}\s*:\s*'
        r'(?P<quote>\\"|")(?P<value>(?:\\(?!")|[^"\\])*)$',
        object_tail,
    )
    if match is None:
        return None
    return match.group("key")

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
            
    #domains = re.findall(r'"domain"\s*:\s*"([^"]*)', payload)
    #intents = re.findall(r'"intent"\s*:\s*"([^"]*)', payload)
    # Capture a string value that may be incomplete.
    # This intentionally stops before normal/escaped closing quotes.
    VAL = r'((?:\\(?!")|[^"\\])*)'

    domains = re.findall(
        rf'{key_re("domain")}\s*:\s*{Q}{VAL}{Q}',
        payload,
    )
    intents = re.findall(
        rf'{key_re("intent")}\s*:\s*{Q}{VAL}{Q}',
        payload,
    )
    cur_domain = domains[-1] if domains else ""
    cur_intent = intents[-1] if intents else ""

    if (
        re.search(key_re("asr_text"), payload)
        and payload.rstrip().endswith('"')
        and not re.search(key_re("semantics"), payload)
    ):
        return DecodingState(STATE_ASR_TEXT, inside_string=inside, json_depth=depth)

    active_domain = _active_value(payload, "domain")
    if active_domain is not None:
        return DecodingState(
            STATE_DOMAIN,
            current_domain=cur_domain,
            current_intent=cur_intent,
            active_label_prefix=active_domain[0],
            active_label_quote=active_domain[1],
            active_label=True,
            inside_string=inside,
            json_depth=depth,
        )

    active_intent = _active_value(payload, "intent")
    if active_intent is not None:
        return DecodingState(
            STATE_INTENT,
            current_domain=cur_domain,
            current_intent=cur_intent,
            active_label_prefix=active_intent[0],
            active_label_quote=active_intent[1],
            active_label=True,
            inside_string=inside,
            json_depth=depth,
        )

    open_object = _latest_open_object(payload)
    if open_object is not None:
        object_name, object_tail = open_object
        key_state = (
            STATE_SLOTS_KEY if object_name == "slots" else STATE_IMPLICIT_SLOTS_KEY
        )
        value_state = (
            STATE_SLOTS_VALUE
            if object_name == "slots"
            else STATE_IMPLICIT_SLOTS_VALUE
        )
        active_value_key = _active_object_value(object_tail)
        if active_value_key is not None:
            return DecodingState(
                value_state,
                current_domain=cur_domain,
                current_intent=cur_intent,
                current_slot_key=active_value_key,
                inside_string=inside,
                json_depth=depth,
            )

        active_key = _active_object_key(object_tail)
        return DecodingState(
            key_state,
            current_domain=cur_domain,
            current_intent=cur_intent,
            active_label_prefix=active_key[0] if active_key else "",
            active_label_quote=active_key[1] if active_key else "",
            active_label=active_key is not None,
            inside_string=inside,
            json_depth=depth,
        )

    if re.search(key_re("semantics"), payload):
        return DecodingState(
            STATE_SEMANTICS,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    return DecodingState(STATE_JSON_PREFIX, inside_string=inside, json_depth=depth)
