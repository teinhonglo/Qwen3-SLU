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
        alpha_domain_intent=0.1,
        alpha_slot_key=0.1,
        grounding_strength=0.1,
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
        self.debug_stats = {
            "steps": 0,
            "state_domain": 0,
            "state_intent": 0,
            "state_slots_key": 0,
            "state_slots_value": 0,
            "di_applied": 0,
            "di_skipped_shape": 0,
            "sk_applied": 0,
            "sk_skipped_shape": 0,
            "changed_max": 0
        }
        print(f"alpha_domain_intent: {self.a_di},  alpha_slot_key: {self.a_sk}, grounding_strength: {self.grounding_strength}")

    def _mask_allowed_strings(self, logits, allowed):
        if not allowed:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        for s in allowed:
            token_ids = self.tok.encode(s, add_special_tokens=False)
            for tid in token_ids[:1]:
                mask[..., tid] = 0.0
        return logits + mask
    
    def _decode_generated_prefix(self, input_ids):
        ids = input_ids[0][self.base_prefix_len :]
        ids = ids.unsqueeze(0)
        if hasattr(self.tok, "batch_decode"):
            return self.tok.batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        if hasattr(self.tok, "tokenizer") and hasattr(self.tok.tokenizer, "batch_decode"):
            return self.tok.tokenizer.batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return self.tok.decode(ids[0], skip_special_tokens=True)
    
    def _decode_top_token_from_logits(self, input_ids, logits):
        if logits is None:
            return None, None

        next_tid = torch.argmax(logits, dim=-1)
        if next_tid.ndim > 0:
            next_tid = next_tid[0]
        next_tid = next_tid.to(input_ids.device).long()

        appended = torch.cat([input_ids[0], next_tid.view(1)], dim=0).unsqueeze(0)
        decoded = self._decode_generated_prefix(appended)
        return next_tid.item(), decoded

    def __call__(self, input_ids, scores):
        #prefix = self.tok.decode(input_ids[0][self.base_prefix_len :], skip_special_tokens=True)
        prefix = self._decode_generated_prefix(input_ids)
        #print("===== PREFIX ESCAPED START =====", flush=True)
        #print(prefix, flush=True)
        #print("===== PREFIX ESCAPED END =====", flush=True)
        #print("===== PREFIX ESCAPED START =====", flush=True)
        #print(prefix, flush=True)
        state = parse_state(prefix)
        #print(prefix)
        #－－print(state, flush=True)
        #print("===== PREFIX ESCAPED END =====", flush=True)
        out = scores
        ori_idx = torch.argmax(out)
        self.debug_stats["steps"] += 1
        if state.state_name == STATE_DOMAIN:
            self.debug_stats["state_domain"] += 1
        elif state.state_name == STATE_INTENT:
            self.debug_stats["state_intent"] += 1
        elif state.state_name == STATE_SLOTS_KEY:
            self.debug_stats["state_slots_key"] += 1
        elif state.state_name == STATE_SLOTS_VALUE:
            self.debug_stats["state_slots_value"] += 1

        print("="*100)
        print(
                f"[DExperts][Ori][state={state.state_name}] decoded={prefix}",
                flush=True,
            )
              
        if state.state_name in (STATE_DOMAIN, STATE_INTENT) and self.di is not None:
            z = self.di.score_next_token(prefix)
            z_tid, z_decoded = self._decode_top_token_from_logits(input_ids, z)
            out_tid, out_decoded = self._decode_top_token_from_logits(input_ids, out)
            if z_tid is not None:
                print(
                    f"[DExperts][DI][state={state.state_name}] out_top_token_id={out_tid}, z_top_token_id={z_tid},\no_decoded={out_decoded}\nz_decoded={z_decoded}",
                    flush=True,
                )
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_di * z.to(out.device)
                self.debug_stats["di_applied"] += 1
            elif z is not None:
                self.debug_stats["di_skipped_shape"] += 1

        if state.state_name == STATE_SLOTS_KEY and self.sk is not None:
            z = self.sk.score_next_token(prefix)
            z_tid, z_decoded = self._decode_top_token_from_logits(input_ids, z)
            out_tid, out_decoded = self._decode_top_token_from_logits(input_ids, out)
            if z_tid is not None:
                print(
                    f"[DExperts][SK][state={state.state_name}] out_top_token_id={out_tid}, z_top_token_id={z_tid},\no_decoded={out_decoded}\nz_decoded={z_decoded}",
                    flush=True,
                )
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_sk * z.to(out.device)
                self.debug_stats["sk_applied"] += 1
            elif z is not None:
                self.debug_stats["sk_skipped_shape"] += 1
        '''
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
        '''

        if self.enable_grounding and state.state_name == STATE_SLOTS_VALUE:
            asr_text = ""
            if '"asr_text"' in prefix:
                try:
                    asr_text = prefix.split('"asr_text"', 1)[1].split('"', 2)[2]
                except Exception:
                    asr_text = ""
            out_tid, out_decoded = self._decode_top_token_from_logits(input_ids, out)
            out = apply_copy_bias(
                out,
                build_copy_bias_map(self.tok, asr_text),
                self.grounding_strength,
            )
            z_tid, z_decoded = self._decode_top_token_from_logits(input_ids, out)
            if z_tid is not None:
                print(
                    f"[DExperts][SV][state={state.state_name}] out_top_token_id={out_tid}, z_top_token_id={z_tid},\no_decoded={out_decoded}\nz_decoded={z_decoded}",
                    flush=True,
                )
        ch_idx = torch.argmax(out)

        if ch_idx != ori_idx:
            self.debug_stats["changed_max"] += 1

        return out
    
    def get_debug_stats(self):
        return dict(self.debug_stats)
