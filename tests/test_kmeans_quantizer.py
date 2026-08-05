import torch
from torch import nn
from finetuning.kmeans_quantizer import ScalarKMeansQuantizer, pack_indices, unpack_indices

class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.q_proj=nn.Linear(8,8); self.norm=nn.LayerNorm(8); self.embed_tokens=nn.Embedding(4,8)

def test_scope_centroids_values_determinism_and_no_nan():
    torch.manual_seed(2); a=Tiny(); b=Tiny(); b.load_state_dict(a.state_dict())
    qa=ScalarKMeansQuantizer(a,4); qb=ScalarKMeansQuantizer(b,4)
    assert qa.selected_names==["q_proj.weight"] and torch.equal(qa.tensors["q_proj.weight"].centroids,qb.tensors["q_proj.weight"].centroids)
    assert qa.tensors["q_proj.weight"].centroids.numel()<=16
    qa.update_and_project(); values=torch.unique(a.q_proj.weight.detach().cpu()); centroids=qa.tensors["q_proj.weight"].centroids
    assert all(torch.any(torch.isclose(v,centroids)) for v in values) and not torch.isnan(centroids).any()

def test_four_and_eight_bit_round_trip():
    for bits, values in [(4,torch.tensor([0,15,2,7,1])),(8,torch.tensor([0,255,2,7]))]:
        assert torch.equal(unpack_indices(pack_indices(values,bits),bits,len(values)),values)

def test_quantizer_resume_state():
    model=Tiny(); q=ScalarKMeansQuantizer(model,4); q.update_and_project(); state=q.state_dict()
    resumed=ScalarKMeansQuantizer(model,4,initialize=False); resumed.load_state_dict(state)
    assert resumed.step_counter==1 and torch.equal(resumed.tensors["q_proj.weight"].assignments,state["tensors"]["q_proj.weight"]["assignments"])
