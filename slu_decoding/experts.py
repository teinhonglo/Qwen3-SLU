import os
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ExpertLM:
    def __init__(self, path, device="cpu"):
        self.path = path
        self.model = None
        self.tokenizer = None
        self.device = device

        if not path or not os.path.isdir(path):
            warnings.warn(f"expert missing: {path}")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path).to(device)
        self.model.eval()

    def score_next_token(self, prefix_text: str):
        if self.model is None:
            return None
        with torch.inference_mode():
            ids = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
            out = self.model(**ids)
            return out.logits[:, -1, :]
