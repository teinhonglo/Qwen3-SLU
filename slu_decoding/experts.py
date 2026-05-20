import os
import json
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from qwen_asr import Qwen3ASRModel
from finetuning.train_expert_lm import TextOnlyExpertConfig, TextOnlyExpertModel

def load_train_conf_from_exp_dir(exp_dir: str):
    if not exp_dir:
        return None

    train_conf_path = os.path.join(exp_dir, "train_conf.json")
    if not os.path.isfile(train_conf_path):
        raise FileNotFoundError(f"train_conf.json not found under exp_dir: {train_conf_path}")

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf.json must be [training_args, model_args]")
    if not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("Both train_conf entries must be dictionaries")
    return cfg

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

        train_conf = load_train_conf_from_exp_dir(self.path)
        if train_conf is None:
            raise ValueError("Unable to load train_conf from exp_dir {self.path}")

        training_args_conf, model_args_conf = train_conf

        base_model_path = model_args_conf["model_path"]
        init_from_asr = bool(model_args_conf.get("init_from_asr", True))

        if init_from_asr:
            print("Init from ASR and LoRA")
            asr_wrapper = Qwen3ASRModel.from_pretrained(base_model_path)
            asr_model = asr_wrapper.model
            thinker = asr_model.thinker
            text_model = thinker.model
            lm_head = thinker.lm_head
            text_cfg = getattr(asr_model.config, "text_config", None)
            hidden_size = getattr(text_cfg, "hidden_size", lm_head.in_features)
            vocab_size = getattr(text_cfg, "vocab_size", lm_head.out_features)
            base_model = TextOnlyExpertModel(
                TextOnlyExpertConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                ),
                text_model=text_model,
                lm_head=lm_head,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        if os.path.isfile(adapter_config_path):
            peft_cfg = PeftConfig.from_pretrained(path)
            self.model = PeftModel.from_pretrained(base_model, path).to(device)
        else:
            self.model = base_model

        self.model.eval()

    def score_next_token(self, prefix_text: str):
        if self.model is None:
            return None
        with torch.inference_mode():
            ids = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
            out = self.model(**ids)
            return out.logits[:, -1, :]
