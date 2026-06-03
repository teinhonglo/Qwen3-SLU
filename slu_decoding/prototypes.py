"""Prototype-based label helpers for MAC-SLU decoding."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SEP = "|||"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object expected: {path}")
    return obj


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0.0:
        return [0.0 for _ in vec]
    return [float(x) / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("-inf")
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


@dataclass
class PrototypeHit:
    label: str
    score: float
    count: int = 0
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"label": self.label, "score": self.score, "count": self.count}
        if self.meta:
            out["meta"] = self.meta
        return out


class MACSLULabelSchema:
    """Valid MAC-SLU labels parsed from labels.txt plus optional train-derived schema."""

    def __init__(self, labels_path: str = "", schema_path: str = ""):
        self.domains = set()
        self.domain2intents: Dict[str, set] = defaultdict(set)
        self.domain2slot_keys: Dict[str, set] = defaultdict(set)
        self.domain_intent2slot_keys: Dict[str, set] = defaultdict(set)
        if labels_path:
            self._load_labels_txt(labels_path)
        if schema_path:
            self._load_schema_json(schema_path)

    def _load_labels_txt(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        section = ""
        current_domain = ""
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("DOMAIN_INTENT_LIST"):
                    section = "domain_intent"
                    current_domain = ""
                    continue
                if stripped.startswith("SLOT_LIST"):
                    section = "slot"
                    current_domain = ""
                    continue
                if stripped.startswith('"""'):
                    continue
                if section == "domain_intent":
                    if not stripped.startswith("-"):
                        continue
                    name = stripped[1:].strip()
                    if not name:
                        continue
                    # Domain lines are written with no leading indentation in labels.txt.
                    if line.startswith("- "):
                        current_domain = name
                        self.domains.add(current_domain)
                    elif current_domain:
                        self.domain2intents[current_domain].add(name)
                elif section == "slot":
                    if not stripped.startswith("-"):
                        continue
                    body = stripped[1:].strip().split(":", 1)[0].strip()
                    if "-" not in body:
                        continue
                    domain, slot_key = body.split("-", 1)
                    domain = domain.strip()
                    slot_key = slot_key.strip()
                    if not domain or not slot_key:
                        continue
                    self.domains.add(domain)
                    self.domain2slot_keys[domain].add(slot_key)

    def _load_schema_json(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        obj = load_json(path)
        for d in obj.get("domains", []) or []:
            if d:
                self.domains.add(str(d))
        for d, intents in (obj.get("domain2intents", {}) or {}).items():
            for intent in intents or []:
                if d and intent:
                    self.domains.add(str(d))
                    self.domain2intents[str(d)].add(str(intent))
        for di_key, slots in (obj.get("domain_intent2slot_keys", {}) or {}).items():
            parts = str(di_key).split(SEP)
            if len(parts) >= 2:
                domain, intent = parts[0], parts[1]
                for slot_key in slots or []:
                    self.add_slot_key(domain, intent, str(slot_key))

    def add_domain_intent(self, domain: str, intent: str) -> None:
        if domain:
            self.domains.add(domain)
        if domain and intent:
            self.domain2intents[domain].add(intent)

    def add_slot_key(self, domain: str, intent: str, slot_key: str) -> None:
        if not domain or not slot_key:
            return
        self.domains.add(domain)
        self.domain2slot_keys[domain].add(slot_key)
        if intent:
            self.domain_intent2slot_keys[f"{domain}{SEP}{intent}"].add(slot_key)

    def valid_domains(self) -> List[str]:
        return sorted(self.domains)

    def valid_intents(self, domain: str = "") -> List[str]:
        if domain:
            return sorted(self.domain2intents.get(domain, set()))
        vals = set()
        for intents in self.domain2intents.values():
            vals.update(intents)
        return sorted(vals)

    def valid_slot_keys(self, domain: str = "", intent: str = "") -> List[str]:
        vals = set()
        if domain and intent:
            vals.update(self.domain_intent2slot_keys.get(f"{domain}{SEP}{intent}", set()))
        if domain:
            vals.update(self.domain2slot_keys.get(domain, set()))
        if not vals:
            for slots in self.domain2slot_keys.values():
                vals.update(slots)
        return sorted(vals)

    def is_valid_domain(self, domain: str) -> bool:
        return not self.domains or domain in self.domains

    def is_valid_intent(self, domain: str, intent: str) -> bool:
        vals = self.valid_intents(domain)
        return not vals or intent in vals

    def is_valid_slot_key(self, domain: str, intent: str, slot_key: str) -> bool:
        vals = self.valid_slot_keys(domain, intent)
        return not vals or slot_key in vals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": self.valid_domains(),
            "domain2intents": {k: sorted(v) for k, v in sorted(self.domain2intents.items())},
            "domain2slot_keys": {k: sorted(v) for k, v in sorted(self.domain2slot_keys.items())},
            "domain_intent2slot_keys": {k: sorted(v) for k, v in sorted(self.domain_intent2slot_keys.items())},
        }


class TokenEmbeddingPrefixEmbedder:
    """Mean-pool Qwen token embeddings for a text prefix."""

    def __init__(self, tokenizer, model, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.embedding = self._get_embedding_module(model)
        self.dim = int(getattr(self.embedding, "embedding_dim", 0) or 0)

    @staticmethod
    def _get_embedding_module(model):
        getter = getattr(model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
            if emb is not None:
                return emb
        base = getattr(model, "base_model", None)
        if base is not None:
            getter = getattr(base, "get_input_embeddings", None)
            if callable(getter):
                emb = getter()
                if emb is not None:
                    return emb
        inner = getattr(model, "model", None)
        if inner is not None:
            getter = getattr(inner, "get_input_embeddings", None)
            if callable(getter):
                emb = getter()
                if emb is not None:
                    return emb
        raise ValueError("Unable to locate model input embeddings")

    def __call__(self, text: str, audio_path: str = "", prompt: str = "") -> List[float]:
        ids = self.tokenizer(text or "", return_tensors="pt", add_special_tokens=False).input_ids
        if ids.numel() == 0:
            ids = torch.tensor([[getattr(self.tokenizer, "eos_token_id", 0) or 0]], dtype=torch.long)
        ids = ids.to(next(self.embedding.parameters()).device)
        with torch.inference_mode():
            vec = self.embedding(ids).float().mean(dim=1)[0]
            vec = F.normalize(vec, dim=0)
        return vec.detach().cpu().tolist()




class AudioStatsPrefixEmbedder:
    """Concatenate token-prefix embeddings with lightweight audio statistics.

    This keeps the v2 audio+prefix path switchable without training any expert LM.
    It intentionally avoids nested model forwards inside logits processors.
    """

    def __init__(self, text_embedder: TokenEmbeddingPrefixEmbedder, sample_rate: int = 16000):
        self.text_embedder = text_embedder
        self.sample_rate = int(sample_rate)
        self.dim = text_embedder.dim + 8

    @staticmethod
    def _audio_stats(audio_path: str, sample_rate: int) -> List[float]:
        if not audio_path or not os.path.isfile(audio_path):
            return [0.0] * 8
        try:
            import librosa
            wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception:
            return [0.0] * 8
        if wav is None or len(wav) == 0:
            return [0.0] * 8
        t = torch.tensor(wav, dtype=torch.float32)
        duration = float(t.numel()) / float(sample_rate)
        mean = float(t.mean())
        std = float(t.std(unbiased=False))
        rms = float(torch.sqrt(torch.mean(t * t)))
        max_abs = float(torch.max(torch.abs(t)))
        zcr = float(((t[1:] * t[:-1]) < 0).float().mean()) if t.numel() > 1 else 0.0
        q25 = float(torch.quantile(torch.abs(t), 0.25))
        q75 = float(torch.quantile(torch.abs(t), 0.75))
        return [duration, mean, std, rms, max_abs, zcr, q25, q75]

    def __call__(self, text: str, audio_path: str = "", prompt: str = "") -> List[float]:
        text_vec = self.text_embedder(text, audio_path=audio_path, prompt=prompt)
        audio_vec = self._audio_stats(audio_path, self.sample_rate)
        return l2_normalize(list(text_vec) + audio_vec)


class PrototypeIndex:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.prototype_source = data.get("prototype_source", "text_prefix")
        self.embedding_backend = data.get("embedding_backend", "token_embedding_mean")
        self.domain = data.get("domain", {}) or {}
        self.intent = data.get("intent", {}) or {}
        self.slot_key = data.get("slot_key", {}) or {}

    @classmethod
    def load(cls, path: str) -> "PrototypeIndex":
        return cls(load_json(path))

    @staticmethod
    def _entries(section: Dict[str, Any]) -> Iterable[Tuple[str, List[float], int, Dict[str, Any]]]:
        for key, item in section.items():
            if not isinstance(item, dict):
                continue
            vec = item.get("vector") or []
            yield key, vec, int(item.get("count", 0) or 0), item.get("meta", {}) or {}

    def search(self, kind: str, query_vec: Sequence[float], top_k: int = 5, allowed: Optional[Iterable[str]] = None, domain: str = "", intent: str = "") -> List[PrototypeHit]:
        section = {"domain": self.domain, "intent": self.intent, "slot_key": self.slot_key}.get(kind, {})
        allowed_set = set(allowed or [])
        hits: List[PrototypeHit] = []
        for key, vec, count, meta in self._entries(section):
            label = meta.get("label", key)
            if allowed_set and label not in allowed_set:
                continue
            if kind == "intent" and domain and meta.get("domain") and meta.get("domain") != domain:
                continue
            if kind == "slot_key":
                if domain and meta.get("domain") and meta.get("domain") != domain:
                    continue
                if intent and meta.get("intent") and meta.get("intent") != intent:
                    # keep domain-level/global fallbacks only if no intent metadata is present.
                    continue
            hits.append(PrototypeHit(label=label, score=cosine(query_vec, vec), count=count, meta=meta))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


def choose_replacement(current: str, hits: Sequence[PrototypeHit], threshold: float, margin: float, current_is_valid: bool) -> Tuple[str, str]:
    if not hits:
        return current, "no_hit"
    top = hits[0]
    second_score = hits[1].score if len(hits) > 1 else float("-inf")
    if top.label == current:
        return current, "top_matches_current"
    effective_threshold = threshold if current_is_valid else min(threshold, 0.0)
    if top.score < effective_threshold:
        return current, "below_threshold"
    if second_score != float("-inf") and (top.score - second_score) < margin and current_is_valid:
        return current, "below_margin"
    return top.label, "replaced" if current_is_valid else "replaced_invalid"


def parse_semantics_field(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            return []
    return []
