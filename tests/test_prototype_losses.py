import torch

from qwen_asr.core.transformers_backend.prototype_losses import (
    LearnableScaledCosine,
    prototype_multi_label_loss,
)


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


def test_prototype_multi_label_loss_uses_bce_logits():
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])

    loss = prototype_multi_label_loss(logits, targets, {"loss_type": "bce"})
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    assert torch.allclose(loss, expected)
