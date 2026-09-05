import torch
from transformers import LogitsProcessor

from .grounding import (
    apply_copy_bias,
    build_copy_bias_map,
    trim_asr_text_left_of_decoded_value,
)
from .state_parser import (
    STATE_COMPLETE,
    STATE_DOMAIN,
    STATE_IMPLICIT_SLOTS_KEY,
    STATE_IMPLICIT_SLOTS_NEXT,
    STATE_INTENT,
    STATE_SLOTS_KEY,
    STATE_SLOTS_NEXT,
    STATE_SLOTS_VALUE,
    parse_state,
)
from .token_trie import TokenIDTrie, remaining_text_candidates

import re

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
        schema_constraint_mode="hard",
        schema_constraint_strength=1.0,
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
        self.schema_constraint_mode = (
            str(schema_constraint_mode).lower() if enable_schema_mask else "off"
        )
        if self.schema_constraint_mode not in ("off", "soft", "hard"):
            raise ValueError(
                "schema_constraint_mode must be one of: off, soft, hard"
            )
        self.schema_constraint_strength = float(schema_constraint_strength)
        self.enable_grounding = enable_grounding
        self._continuation_trie_cache = {}
        self.debug_stats = {}
        self.reset()
        print(
            "DExperts/schema decoding: "
            f"alpha_domain_intent={self.a_di}, alpha_slot_key={self.a_sk}, "
            f"schema_mode={self.schema_constraint_mode}, "
            f"schema_strength={self.schema_constraint_strength}, "
            f"grounding_strength={self.grounding_strength}"
        )

    def reset(self):
        self.debug_stats = {
            "steps": 0,
            "state_domain": 0,
            "state_intent": 0,
            "state_slots_key": 0,
            "state_implicit_slots_key": 0,
            "state_slots_value": 0,
            "di_applied": 0,
            "di_skipped_shape": 0,
            "sk_applied": 0,
            "sk_skipped_shape": 0,
            "schema_applied": 0,
            "schema_no_candidates": 0,
            "schema_prefix_miss": 0,
            "structure_applied": 0,
            "structure_prefix_miss": 0,
            "eos_forced": 0,
            "changed_max": 0,
        }

    def _schema_allowed_strings(self, state):
        if self.schema is None:
            return []
        if state.state_name == STATE_DOMAIN:
            return self.schema.get_valid_domains()
        if state.state_name == STATE_INTENT:
            return self.schema.get_valid_intents(state.current_domain)
        if state.state_name == STATE_SLOTS_KEY:
            return self.schema.get_valid_slot_keys(
                state.current_domain, state.current_intent
            )
        if state.state_name == STATE_IMPLICIT_SLOTS_KEY:
            return self.schema.get_valid_implicit_slot_keys(
                state.current_domain, state.current_intent
            )
        return []

    def _encode(self, text):
        return [
            int(token_id)
            for token_id in self.tok.encode(text, add_special_tokens=False)
        ]

    def _get_continuation_trie(self, continuations):
        cache_key = tuple(continuations)
        cached = self._continuation_trie_cache.get(cache_key)
        if cached is not None:
            return cached

        trie = TokenIDTrie()
        for continuation in continuations:
            trie.insert(self._encode(continuation))
        self._continuation_trie_cache[cache_key] = trie
        return trie

    def _next_token_ids_for_continuations(self, continuations):
        continuations = tuple(dict.fromkeys(text for text in continuations if text))
        if not continuations:
            return None
        trie = self._get_continuation_trie(continuations)
        return trie.next_token_ids([])

    def _schema_followup(self, state):
        quote = state.active_label_quote
        if state.state_name == STATE_DOMAIN:
            return f"{quote}, {quote}intent{quote}: {quote}"
        if state.state_name == STATE_INTENT:
            return f"{quote}, {quote}slots{quote}: {{"
        if state.state_name in (STATE_SLOTS_KEY, STATE_IMPLICIT_SLOTS_KEY):
            return f"{quote}: {quote}"
        return quote

    def _schema_next_token_ids(self, state, allowed):
        prefix = state.active_label_prefix
        followup = self._schema_followup(state)
        candidates = [f"{label}{followup}" for label in allowed]
        continuations = remaining_text_candidates(candidates, prefix)
        return self._next_token_ids_for_continuations(continuations)

    def _structure_allowed_strings(self, state):
        allowed = list(state.structure_candidates)
        if state.state_name == STATE_SLOTS_NEXT and self.schema is not None:
            if not self.schema.get_valid_slot_keys(
                state.current_domain, state.current_intent
            ):
                allowed = [literal for literal in allowed if literal == "}"]
        elif (
            state.state_name == STATE_IMPLICIT_SLOTS_NEXT
            and self.schema is not None
        ):
            if not self.schema.get_valid_implicit_slot_keys(
                state.current_domain, state.current_intent
            ):
                allowed = [literal for literal in allowed if literal == "}"]
        return allowed

    def _structure_next_token_ids(self, state, allowed):
        continuations = remaining_text_candidates(
            allowed, state.active_structure_prefix
        )
        return self._next_token_ids_for_continuations(continuations)

    def _apply_allowed_token_ids(self, logits, valid_ids):
        if self.schema_constraint_mode == "soft":
            constrained = logits.clone()
            constrained[..., valid_ids] += self.schema_constraint_strength
            return constrained

        constrained = torch.full_like(logits, float("-inf"))
        constrained[..., valid_ids] = logits[..., valid_ids]
        return constrained

    def _apply_schema_constraint(self, logits, input_ids, state):
        if self.schema_constraint_mode == "off":
            return logits

        if state.state_name == STATE_COMPLETE:
            eos_token_id = getattr(self.tok, "eos_token_id", None)
            if isinstance(eos_token_id, (tuple, list)):
                eos_token_ids = [int(token_id) for token_id in eos_token_id]
            elif eos_token_id is None:
                eos_token_ids = []
            else:
                eos_token_ids = [int(eos_token_id)]
            valid_ids = [
                token_id
                for token_id in eos_token_ids
                if 0 <= token_id < logits.shape[-1]
            ]
            if not valid_ids:
                return logits
            self.debug_stats["eos_forced"] += 1
            return self._apply_allowed_token_ids(logits, valid_ids)

        if state.active_structure:
            allowed = self._structure_allowed_strings(state)
            next_ids = self._structure_next_token_ids(state, allowed)
            if next_ids is None or not next_ids:
                self.debug_stats["structure_prefix_miss"] += 1
                return logits
            valid_ids = [tid for tid in next_ids if 0 <= tid < logits.shape[-1]]
            if not valid_ids:
                self.debug_stats["structure_prefix_miss"] += 1
                return logits
            self.debug_stats["structure_applied"] += 1
            return self._apply_allowed_token_ids(logits, valid_ids)

        if self.schema is None or not state.active_label:
            return logits

        allowed = self._schema_allowed_strings(state)
        if not allowed:
            self.debug_stats["schema_no_candidates"] += 1
            return logits

        next_ids = self._schema_next_token_ids(state, allowed)
        if next_ids is None or not next_ids:
            # A generated prefix cannot be repaired by a next-token mask.  Keep
            # base decoding alive instead of turning every logit into -inf.
            self.debug_stats["schema_prefix_miss"] += 1
            return logits

        valid_ids = [tid for tid in next_ids if 0 <= tid < logits.shape[-1]]
        if not valid_ids:
            self.debug_stats["schema_prefix_miss"] += 1
            return logits

        self.debug_stats["schema_applied"] += 1
        return self._apply_allowed_token_ids(logits, valid_ids)
    
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
        elif state.state_name == STATE_IMPLICIT_SLOTS_KEY:
            self.debug_stats["state_implicit_slots_key"] += 1
        elif state.state_name == STATE_SLOTS_VALUE:
            self.debug_stats["state_slots_value"] += 1

        if (
            state.active_label
            and state.state_name in (STATE_DOMAIN, STATE_INTENT)
            and self.di is not None
            and self.a_di != 0.0
        ):
            z = self.di.score_next_token(prefix)
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_di * z.to(out.device)
                self.debug_stats["di_applied"] += 1
            elif z is not None:
                self.debug_stats["di_skipped_shape"] += 1

        if (
            state.active_label
            and state.state_name == STATE_SLOTS_KEY
            and self.sk is not None
            and self.a_sk != 0.0
        ):
            z = self.sk.score_next_token(prefix)
            if z is not None and z.shape[-1] == out.shape[-1]:
                out = out + self.a_sk * z.to(out.device)
                self.debug_stats["sk_applied"] += 1
            elif z is not None:
                self.debug_stats["sk_skipped_shape"] += 1
        out = self._apply_schema_constraint(out, input_ids, state)

        if self.enable_grounding and state.state_name == STATE_SLOTS_VALUE:
            asr_text = ""
            if '"asr_text"' in prefix:
                try:
                    pattern = r'<asr_text>\{\s*"asr_text"\s*:\s*"(?P<asr_text>(?:\\.|[^"\\])*)"'
                    m = re.search(pattern, prefix)
                    asr_text = m.group("asr_text")
                except Exception:
                    asr_text = ""
            asr_text = trim_asr_text_left_of_decoded_value(asr_text, prefix)
            out_tid, out_decoded = self._decode_top_token_from_logits(input_ids, out)
            out = apply_copy_bias(
                out,
                build_copy_bias_map(self.tok, asr_text),
                self.grounding_strength,
            )
            z_tid, z_decoded = self._decode_top_token_from_logits(input_ids, out)
            if z_tid is not None:
                print(
                    f"[DExperts][SV][state={state.state_name}] asr_text={asr_text}, out_top_token_id={out_tid}, z_top_token_id={z_tid},\no_decoded={out_decoded}\nz_decoded={z_decoded}",
                    flush=True,
                )
        ch_idx = torch.argmax(out)

        if ch_idx != ori_idx:
            self.debug_stats["changed_max"] += 1

        return out
    
    def get_debug_stats(self):
        return dict(self.debug_stats)


class StateAwarePrototypeTrackerLogitsProcessor(LogitsProcessor):
    """Track prototype label decisions at semantic decoding states.

    This processor intentionally leaves logits unchanged by default. It mirrors the
    DExperts state-tracking path, but records nearest prototype labels so the final
    structured semantics can be repaired after generation.
    """

    def __init__(
        self,
        tokenizer,
        prototype_index,
        embed_text_fn,
        label_schema=None,
        base_prefix_len=0,
        top_k=5,
        min_step_gap=2,
        log_prefix="Prototype",
    ):
        self.tok = tokenizer
        self.prototype_index = prototype_index
        self.embed_text_fn = embed_text_fn
        self.label_schema = label_schema
        self.base_prefix_len = base_prefix_len
        self.top_k = int(top_k)
        self.min_step_gap = int(min_step_gap)
        self.log_prefix = log_prefix
        self.records = []
        self.debug_stats = {
            "steps": 0,
            "state_domain": 0,
            "state_intent": 0,
            "state_slots_key": 0,
            "state_slots_value": 0,
            "prototype_domain": 0,
            "prototype_intent": 0,
            "prototype_slot_key": 0,
        }
        self._last_record_step = {}

    def reset(self):
        self.records = []
        self._last_record_step = {}
        for key in self.debug_stats:
            self.debug_stats[key] = 0

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

    def _maybe_record(self, kind, state, prefix):
        step = self.debug_stats["steps"]
        last = self._last_record_step.get(kind, -10**9)
        if step - last < self.min_step_gap:
            return
        self._last_record_step[kind] = step

        try:
            qvec = self.embed_text_fn(prefix)
        except Exception as exc:
            self.records.append({"kind": kind, "state": state.state_name, "error": str(exc), "step": step})
            return

        allowed = None
        domain = state.current_domain or ""
        intent = state.current_intent or ""
        if self.label_schema is not None:
            if kind == "domain":
                allowed = self.label_schema.valid_domains()
            elif kind == "intent":
                allowed = self.label_schema.valid_intents(domain)
            elif kind == "slot_key":
                allowed = self.label_schema.valid_slot_keys(domain, intent)

        hits = self.prototype_index.search(
            kind,
            qvec,
            top_k=self.top_k,
            allowed=allowed,
            domain=domain,
            intent=intent,
        )
        self.records.append(
            {
                "kind": kind,
                "state": state.state_name,
                "step": step,
                "current_domain": domain,
                "current_intent": intent,
                "top": [h.to_dict() for h in hits],
                "prefix_tail": prefix[-300:],
            }
        )
        self.debug_stats[f"prototype_{kind}"] += 1
        if hits:
            print(
                f"[{self.log_prefix}][{kind}] state={state.state_name} top={hits[0].label} score={hits[0].score:.4f}",
                flush=True,
            )

    def __call__(self, input_ids, scores):
        prefix = self._decode_generated_prefix(input_ids)
        state = parse_state(prefix)
        self.debug_stats["steps"] += 1
        if state.state_name == STATE_DOMAIN:
            self.debug_stats["state_domain"] += 1
            self._maybe_record("domain", state, prefix)
        elif state.state_name == STATE_INTENT:
            self.debug_stats["state_intent"] += 1
            self._maybe_record("intent", state, prefix)
        elif state.state_name == STATE_SLOTS_KEY:
            self.debug_stats["state_slots_key"] += 1
            self._maybe_record("slot_key", state, prefix)
        elif state.state_name == STATE_SLOTS_VALUE:
            self.debug_stats["state_slots_value"] += 1
        return scores

    def get_debug_stats(self):
        return dict(self.debug_stats)

    def get_records(self):
        return list(self.records)
