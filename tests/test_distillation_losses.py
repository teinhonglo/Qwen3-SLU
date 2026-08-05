import pytest
import torch
from finetuning.distillation_losses import (RepresentationProjector, masked_mean_pool,
    masked_token_kl, representation_contrastive_loss, validate_logit_compatibility)

def test_kl_masking_and_temperature_scaling():
    torch.manual_seed(1); s=torch.randn(2,3,5); t=torch.randn(2,3,5); labels=torch.tensor([[1,-100,2],[3,4,-100]])
    loss=masked_token_kl(s,t,labels,2.0); tprob=torch.softmax(t/2, -1)
    manual=(tprob*(torch.log_softmax(t/2,-1)-torch.log_softmax(s/2,-1))).sum(-1)[labels!=-100].mean()*4
    assert torch.allclose(loss,manual)

def test_vocabulary_and_sequence_compatibility():
    with pytest.raises(ValueError, match="vocabulary"): validate_logit_compatibility(torch.zeros(1,2,3),torch.zeros(1,2,4),torch.zeros(1,2))
    with pytest.raises(ValueError, match="sequence"): validate_logit_compatibility(torch.zeros(1,3,4),torch.zeros(1,3,4),torch.zeros(1,2))

def test_masked_pool_projection_and_bb_similarity():
    labels=torch.tensor([[1,-100,2],[-100,3,4]]); s=torch.arange(24.).reshape(2,3,4); t=torch.arange(36.).reshape(2,3,6)
    pooled=masked_mean_pool(s,labels); assert torch.equal(pooled[0],(s[0,0]+s[0,2])/2)
    projector=RepresentationProjector([4],[6]); loss, matrices=representation_contrastive_loss([s],[t],labels,[0],[0],projector,1.,return_similarities=True)
    assert loss.ndim==0 and matrices[0].shape==(2,2) and isinstance(projector.student[0],torch.nn.Linear)

def test_batch_size_one_validation():
    labels=torch.tensor([[1,2]]); p=RepresentationProjector([3],[3])
    with pytest.raises(ValueError,match="batch size > 1"): representation_contrastive_loss([torch.ones(1,2,3)],[torch.ones(1,2,3)],labels,[0],[0],p,1.)
