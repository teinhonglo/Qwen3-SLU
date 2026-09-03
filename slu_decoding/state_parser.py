from dataclasses import dataclass
import json
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
STATE_SEMANTICS_START = "STATE_SEMANTICS_START"
STATE_AFTER_DOMAIN = "STATE_AFTER_DOMAIN"
STATE_AFTER_INTENT = "STATE_AFTER_INTENT"
STATE_SLOTS_NEXT = "STATE_SLOTS_NEXT"
STATE_AFTER_SLOT_KEY = "STATE_AFTER_SLOT_KEY"
STATE_AFTER_SLOTS = "STATE_AFTER_SLOTS"
STATE_IMPLICIT_SLOTS_NEXT = "STATE_IMPLICIT_SLOTS_NEXT"
STATE_AFTER_IMPLICIT_SLOT_KEY = "STATE_AFTER_IMPLICIT_SLOT_KEY"
STATE_AFTER_IMPLICIT_SLOTS = "STATE_AFTER_IMPLICIT_SLOTS"
STATE_COMPLETE = "STATE_COMPLETE"


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
    active_structure_prefix: str = ""
    structure_candidates: tuple = ()
    active_structure: bool = False

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


def _completed_values(text: str, key_name: str):
    """Return completed string values together with their quote and end offset."""
    pattern = (
        rf'{key_re(key_name)}\s*:\s*'
        rf'(?P<quote>\\"|")(?P<value>(?:\\(?!")|[^"\\])*)(?P=quote)'
    )
    return [
        (match.group("value"), match.group("quote"), match.end())
        for match in re.finditer(pattern, text)
    ]


def _structure_state(
    state_name: str,
    prefix: str,
    candidates,
    current_domain: str = "",
    current_intent: str = "",
    inside_string: bool = False,
    json_depth: int = 0,
):
    candidates = tuple(candidates)
    if any(candidate.startswith(prefix) for candidate in candidates):
        return DecodingState(
            state_name,
            current_domain=current_domain,
            current_intent=current_intent,
            inside_string=inside_string,
            json_depth=json_depth,
            active_structure_prefix=prefix,
            structure_candidates=candidates,
            active_structure=True,
        )
    return None


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


def _latest_closed_object(text: str):
    """Return the latest closed ``slots``/``implicit_slots`` object and suffix."""
    matches = []
    for key_name in ("slots", "implicit_slots"):
        for match in re.finditer(rf'{key_re(key_name)}\s*:\s*\{{', text):
            matches.append((match.start(), match.end(), key_name))
    if not matches:
        return None

    _, end, key_name = max(matches, key=lambda item: item[0])
    close_offset = text[end:].find("}")
    if close_offset < 0:
        return None
    close_end = end + close_offset + 1
    return key_name, text[close_end:]


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


def _latest_completed_object_key(object_tail: str):
    pattern = (
        r'(?:^|,)\s*(?P<quote>\\"|")'
        r'(?P<key>(?:\\(?!")|[^"\\])*)(?P=quote)'
    )
    matches = list(re.finditer(pattern, object_tail))
    if not matches:
        return None
    match = matches[-1]
    return match.group("key"), match.group("quote"), object_tail[match.end():]


def _latest_completed_object_pair(object_tail: str):
    pattern = (
        r'(?:^|,)\s*(?P<key_quote>\\"|")'
        r'(?P<key>(?:\\(?!")|[^"\\])*)(?P=key_quote)\s*:\s*'
        r'(?P<value_quote>\\"|")'
        r'(?P<value>(?:\\(?!")|[^"\\])*)(?P=value_quote)'
    )
    matches = list(re.finditer(pattern, object_tail))
    if not matches:
        return None
    match = matches[-1]
    return (
        match.group("key"),
        match.group("key_quote"),
        object_tail[match.end():],
    )


def _semantics_string_tail(text: str):
    """Return the unfinished outer ``semantics`` string content, if present."""
    match = re.search(
        rf'{key_re("semantics")}\s*:\s*"(?P<tail>.*)$',
        text,
    )
    return match.group("tail") if match is not None else None

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

    try:
        completed_payload = json.loads(payload)
    except Exception:
        completed_payload = None
    if isinstance(completed_payload, dict) and "semantics" in completed_payload:
        return DecodingState(
            STATE_COMPLETE,
            inside_string=inside,
            json_depth=depth,
        )
            
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
        if active_key is not None:
            return DecodingState(
                key_state,
                current_domain=cur_domain,
                current_intent=cur_intent,
                active_label_prefix=active_key[0],
                active_label_quote=active_key[1],
                active_label=True,
                inside_string=inside,
                json_depth=depth,
            )

        completed_pair = _latest_completed_object_pair(object_tail)
        if completed_pair is not None:
            _, quote, tail = completed_pair
            next_state = (
                STATE_SLOTS_NEXT
                if object_name == "slots"
                else STATE_IMPLICIT_SLOTS_NEXT
            )
            structure = _structure_state(
                next_state,
                tail,
                (f", {quote}", "}"),
                cur_domain,
                cur_intent,
                inside,
                depth,
            )
            if structure is not None:
                return structure

        completed_key = _latest_completed_object_key(object_tail)
        if completed_key is not None:
            _, quote, tail = completed_key
            after_key_state = (
                STATE_AFTER_SLOT_KEY
                if object_name == "slots"
                else STATE_AFTER_IMPLICIT_SLOT_KEY
            )
            structure = _structure_state(
                after_key_state,
                tail,
                (f": {quote}",),
                cur_domain,
                cur_intent,
                inside,
                depth,
            )
            if structure is not None:
                return structure

        next_state = (
            STATE_SLOTS_NEXT
            if object_name == "slots"
            else STATE_IMPLICIT_SLOTS_NEXT
        )
        default_quote = '\\"' if '\\"' in payload else '"'
        structure = _structure_state(
            next_state,
            object_tail,
            (default_quote, "}"),
            cur_domain,
            cur_intent,
            inside,
            depth,
        )
        if structure is not None:
            return structure

        return DecodingState(
            key_state,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    completed_intents = _completed_values(payload, "intent")
    if completed_intents:
        _, quote, end = completed_intents[-1]
        structure = _structure_state(
            STATE_AFTER_INTENT,
            payload[end:],
            (f", {quote}slots{quote}: {{",),
            cur_domain,
            cur_intent,
            inside,
            depth,
        )
        if structure is not None:
            return structure

    completed_domains = _completed_values(payload, "domain")
    if completed_domains:
        _, quote, end = completed_domains[-1]
        structure = _structure_state(
            STATE_AFTER_DOMAIN,
            payload[end:],
            (f", {quote}intent{quote}: {quote}",),
            cur_domain,
            cur_intent,
            inside,
            depth,
        )
        if structure is not None:
            return structure

    closed_object = _latest_closed_object(payload)
    if closed_object is not None:
        object_name, tail = closed_object
        if object_name == "slots":
            quote = '\\"' if '\\"' in payload else '"'
            structure = _structure_state(
                STATE_AFTER_SLOTS,
                tail,
                (f", {quote}implicit_slots{quote}: {{",),
                cur_domain,
                cur_intent,
                inside,
                depth,
            )
        else:
            quote = '\\"' if '\\"' in payload else '"'
            final_suffix = '}]"}' if quote == '\\"' else "}]}"
            structure = _structure_state(
                STATE_AFTER_IMPLICIT_SLOTS,
                tail,
                (
                    final_suffix,
                    f"}}, {{{quote}domain{quote}: {quote}",
                ),
                cur_domain,
                cur_intent,
                inside,
                depth,
            )
        if structure is not None:
            return structure

    semantics_tail = _semantics_string_tail(payload)
    if semantics_tail is not None:
        structure = _structure_state(
            STATE_SEMANTICS_START,
            semantics_tail,
            ('[]"}', '[{\\"domain\\": \\"'),
            cur_domain,
            cur_intent,
            inside,
            depth,
        )
        if structure is not None:
            return structure

    if re.search(key_re("semantics"), payload):
        return DecodingState(
            STATE_SEMANTICS,
            current_domain=cur_domain,
            current_intent=cur_intent,
            inside_string=inside,
            json_depth=depth,
        )

    return DecodingState(STATE_JSON_PREFIX, inside_string=inside, json_depth=depth)
