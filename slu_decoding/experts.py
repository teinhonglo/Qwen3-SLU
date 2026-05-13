import os
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


class ExpertLM:
    def __init__(self, path, device="cpu"):
        self.path = path
        self.model = None
        self.tokenizer = None
        self.device = device

        if not path or not os.path.isdir(path):
            warnings.warn(f"expert missing: {path}")
            return

        adapter_config_path = os.path.join(path, "adapter_config.json")
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        if os.path.isfile(adapter_config_path):
            peft_cfg = PeftConfig.from_pretrained(path)
            base_model_path = peft_cfg.base_model_name_or_path
            if not base_model_path:
                raise ValueError(
                    f"Invalid adapter_config.json under {path}: base_model_name_or_path is empty. "
                    "Please retrain expert with a valid --model_name_or_path or set model_path in train_conf."
                )
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
            self.model = PeftModel.from_pretrained(base_model, path).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path).to(device)
        self.model.eval()

    def score_next_token(self, prefix_text: str):
        if self.model is None:
            return None
        with torch.inference_mode():
            ids = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
            out = self.model(**ids)
            return out.logits[:, -1, :]
