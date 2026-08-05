import json
from types import SimpleNamespace

import torch
from torch import nn

from finetuning.vocabulary_pruning import (
    VocabularyRemappingCollator,
    apply_structural_vocabulary_pruning,
    load_vocabulary_mapping,
    remap_token_tensor,
    restore_token_tensor,
)


def test_load_manifest_builds_bidirectional_compact_mapping(tmp_path):
    manifest = tmp_path / "vocabulary.json"
    manifest.write_text(json.dumps({"retained_token_ids": [4, 1, 4, 2]}))
    mapping = load_vocabulary_mapping(str(manifest))
    assert mapping.retained_token_ids == (1, 2, 4)
    assert mapping.old_to_new == {1: 0, 2: 1, 4: 2}
    assert mapping.new_to_old == {0: 1, 1: 2, 2: 4}


def test_token_remapping_round_trip_and_ignored_labels(tmp_path):
    manifest = tmp_path / "vocabulary.json"
    manifest.write_text(json.dumps({"retained_token_ids": [1, 4, 7]}))
    mapping = load_vocabulary_mapping(str(manifest))
    original = torch.tensor([[1, 7, 4]])
    compact = remap_token_tensor(original, mapping)
    assert torch.equal(compact, torch.tensor([[0, 2, 1]]))
    assert torch.equal(restore_token_tensor(compact, mapping), original)
    labels = remap_token_tensor(torch.tensor([[-100, 4, 7]]), mapping, ignore_index=-100)
    assert torch.equal(labels, torch.tensor([[-100, 1, 2]]))


class ToyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 3)
        self.lm_head = nn.Linear(3, 8, bias=False)
        text_config = SimpleNamespace(vocab_size=8, bos_token_id=1, eos_token_id=7, pad_token_id=0)
        self.config = SimpleNamespace(text_config=text_config, vocab_size=8)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = ToyThinker()
        self.config = self.thinker.config


def test_structural_pruning_shrinks_embedding_and_lm_head(tmp_path):
    manifest = tmp_path / "vocabulary.json"
    manifest.write_text(json.dumps({"retained_token_ids": [0, 1, 4, 7]}))
    mapping = load_vocabulary_mapping(str(manifest))
    model = ToyModel()
    embedding_rows = model.thinker.model.embed_tokens.weight.detach().clone()[[0, 1, 4, 7]]
    head_rows = model.thinker.lm_head.weight.detach().clone()[[0, 1, 4, 7]]
    apply_structural_vocabulary_pruning(model, mapping)
    assert model.thinker.model.embed_tokens.num_embeddings == 4
    assert model.thinker.lm_head.out_features == 4
    assert model.thinker.config.text_config.vocab_size == 4
    assert torch.equal(model.thinker.model.embed_tokens.weight, embedding_rows)
    assert torch.equal(model.thinker.lm_head.weight, head_rows)


def test_structural_pruning_preserves_tied_embeddings(tmp_path):
    manifest = tmp_path / "vocabulary.json"
    manifest.write_text(json.dumps({"retained_token_ids": [0, 1, 4, 7]}))
    model = ToyModel()
    model.thinker.lm_head.weight = model.thinker.model.embed_tokens.weight
    apply_structural_vocabulary_pruning(model, load_vocabulary_mapping(str(manifest)))
    assert model.thinker.lm_head.weight is model.thinker.model.embed_tokens.weight


def test_remapping_collator_compacts_inputs_and_labels(tmp_path):
    manifest = tmp_path / "vocabulary.json"
    manifest.write_text(json.dumps({"retained_token_ids": [1, 4, 7]}))
    mapping = load_vocabulary_mapping(str(manifest))

    def collator(_):
        return {"input_ids": torch.tensor([[1, 4, 7]]),
                "labels": torch.tensor([[-100, 4, 7]])}

    batch = VocabularyRemappingCollator(collator, mapping)([{}])
    assert torch.equal(batch["input_ids"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 1, 2]]))
