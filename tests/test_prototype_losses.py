import torch

from qwen_asr.core.transformers_backend.prototype_losses import (
    LearnableScaledCosine,
    clipped_dynamic_margin_loss,
    prototype_multi_label_loss,
)


def test_clipped_dynamic_margin_ignores_easy_pairs():
    similarities = torch.tensor([[0.8, 0.6, -0.2]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])

    loss = clipped_dynamic_margin_loss(similarities, targets, gamma_min=0.1, gamma_max=0.3)

    assert loss.item() == 0.0


def test_clipped_dynamic_margin_reverses_ambiguous_pair_gradient():
    similarities = torch.tensor([[0.55, 0.50]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])

    loss = clipped_dynamic_margin_loss(similarities, targets, gamma_min=0.1, gamma_max=0.3)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.15))
    assert torch.allclose(similarities.grad, torch.tensor([[1.0, -1.0]]))


def test_clipped_dynamic_margin_detaches_hard_pair_margin():
    similarities = torch.tensor([[0.3, 0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])

    loss = clipped_dynamic_margin_loss(similarities, targets, gamma_min=0.1, gamma_max=0.3)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.4))
    assert torch.allclose(similarities.grad, torch.tensor([[-1.0, 1.0]]))


def test_prototype_loss_recovers_cosine_before_dynamic_margin():
    logits = torch.tensor([[5.5, 5.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    config = {
        "loss_type": "clipped_dynamic_margin",
        "normalize": True,
        "temperature": 0.1,
        "dynamic_margin_min": 0.1,
        "dynamic_margin_max": 0.3,
    }

    loss = prototype_multi_label_loss(logits, targets, config)

    assert torch.isclose(loss, torch.tensor(0.15))


def test_scaled_cosine_bce_uses_learnable_scale_and_bias():
    calibration = LearnableScaledCosine(
        num_labels=4,
        scale_init=10.0,
        scale_max=100.0,
    )
    similarities = torch.tensor([[-1.0, 0.0, 1.0]], requires_grad=True)

    logits = calibration(similarities)
    expected_bias = -torch.log(torch.tensor(3.0))

    assert isinstance(calibration.logit_scale, torch.nn.Parameter)
    assert isinstance(calibration.logit_bias, torch.nn.Parameter)
    assert torch.allclose(calibration.scale(), torch.tensor(10.0))
    assert torch.allclose(calibration.logit_bias, expected_bias)
    assert torch.allclose(logits, similarities * 10.0 + expected_bias)
    assert torch.allclose(calibration.recover_similarities(logits), similarities)

    logits.sum().backward()
    assert calibration.logit_scale.grad is not None
    assert calibration.logit_bias.grad is not None
