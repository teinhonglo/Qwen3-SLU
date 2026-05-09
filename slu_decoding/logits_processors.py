import torch
from transformers import LogitsProcessor

from .grounding import apply_copy_bias, build_copy_bias_map
from .state_parser import (
    STATE_DOMAIN,
    STATE_INTENT,
    STATE_SLOTS_KEY,
    STATE_SLOTS_VALUE,
    parse_state,
)


class StateAwareDExpertsLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        base_prefix_len=0,
        schema=None,
        domain_intent_expert=None,
        slot_key_expert=None,
        alpha_domain_intent=1.0,
        alpha_slot_key=1.0,
        grounding_strength=1.0,
        enable_schema_mask=True,
        enable_grounding=True,
    ):
        self.tok = tokenizer
        self.base_prefix_len = base_prefix_len
        self.schema = schema
        self.di = domain_intent_expert
        self.sk = slot_key_expert
        self.a_di = alpha_domain_intent
        self.a_sk = alpha_slot_key
        self.grounding_strength = grounding_strength
        self.enable_schema_mask = enable_schema_mask
        self.enable_grounding = enable_grounding

    def _mask_allowed_strings(self, logits, allowed):
        if not allowed:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        for s in allowed:
            token_ids = self.tok.encode(s, add_special_tokens=False)
            for tid in token_ids[:1]:
                mask[..., tid] = 0.0
        return logits + mask

    def __call__(self, input_ids, scores):
        prefix = self.tok.decode(input_ids[0][self.base_prefix_len :], skip_special_tokens=True)
        state = parse_state(prefix)
        out = scores

        if state.state_name in (STATE_DOMAIN, STATE_INTENT) and self.di is not None:
            z = self.di.score_next_token(prefix)
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_di * z.to(out.device)

        if state.state_name == STATE_SLOTS_KEY and self.sk is not None:
            z = self.sk.score_next_token(prefix)
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_sk * z.to(out.device)

        if self.enable_schema_mask and self.schema is not None:
            if state.state_name == STATE_DOMAIN:
                out = self._mask_allowed_strings(out, self.schema.get_valid_domains())
            elif state.state_name == STATE_INTENT:
                out = self._mask_allowed_strings(
                    out, self.schema.get_valid_intents(state.current_domain)
                )
            elif state.state_name == STATE_SLOTS_KEY:
                out = self._mask_allowed_strings(
                    out,
                    self.schema.get_valid_slot_keys(state.current_domain, state.current_intent),
                )

        if self.enable_grounding and state.state_name == STATE_SLOTS_VALUE:
            asr_text = ""
            if '"asr_text"' in prefix:
                try:
                    asr_text = prefix.split('"asr_text"', 1)[1].split('"', 2)[2]
                except Exception:
                    asr_text = ""
            out = apply_copy_bias(
                out,
                build_copy_bias_map(self.tok, asr_text),
                self.grounding_strength,
            )

        return out
