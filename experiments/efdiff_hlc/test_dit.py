import copy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dit_model import PrithviCrossDiT, AxialRoPE2D, HeadRMSNorm, SwiGLU
from vscore_diffusion import VScoreDiffusion


def tiny(checkpointing=False):
    return PrithviCrossDiT(image_size=16, patch_size=4, hidden_dim=32, depth=2,
                          heads=4, context_dim=24, gradient_checkpointing=checkpointing)


def test_swiglu_gate_and_parameter_budget():
    model = SwiGLU(1024)
    assert model.hidden_dim == 2752
    assert abs(sum(p.numel() for p in model.parameters()) / (2*1024*4096) - 1) < .01
    small = SwiGLU(8, multiple=4)
    with torch.no_grad():
        small.gate_up.weight[:small.hidden_dim].zero_()
    assert torch.count_nonzero(small(torch.randn(2,3,8))) == 0


def test_rope_norm_position_and_relative_displacements():
    rope = AxialRoPE2D(4, 8)
    torch.manual_seed(13)
    q = torch.randn(1,2,1,8).expand(-1,-1,16,-1)
    k = torch.randn(1,2,1,8).expand(-1,-1,16,-1)
    rq,rk=rope(q),rope(k)
    torch.testing.assert_close(rq.square().sum(-1),q.square().sum(-1))
    assert not torch.allclose(rq[:,:,0],rq[:,:,1])
    # Both pairs have displacement (row=0,col=1).
    torch.testing.assert_close((rq[:,:,0]*rk[:,:,1]).sum(-1),(rq[:,:,5]*rk[:,:,6]).sum(-1))
    # Rotating both vectors at the same position preserves their dot product.
    torch.testing.assert_close((rq*rk).sum(-1),(q*k).sum(-1))


def test_qk_norm_scaling_and_bfloat16_finiteness():
    norm=HeadRMSNorm(8)
    x=torch.randn(2,4,16,8)*10
    torch.testing.assert_close(norm(x),norm(x*100),atol=2e-6,rtol=2e-6)
    out=norm((x*1000).bfloat16())
    assert out.dtype==torch.bfloat16 and torch.isfinite(out).all()


def test_optional_legacy_blocks_remain_available():
    model=PrithviCrossDiT(image_size=16,patch_size=4,hidden_dim=32,depth=1,heads=4,
                         context_dim=24,ffn_type='gelu',qk_norm=False,rope_2d=False)
    assert model.pos_embed is not None
    assert model(torch.randn(1,2,16,16),torch.tensor([.5]),torch.randn(1,4,24)).shape==(1,1,16,16)


def activate(model):
    # Zero-initialized head/gates intentionally block deep gradients on update 1.
    with torch.no_grad():
        model.head.weight.normal_(std=.03)
        for block in model.blocks:
            block.modulation[-1].bias.normal_(std=.05)
            block.modulation[-1].weight.normal_(std=.03)


def test_unpatchify_spatial_order():
    model = tiny()
    image = torch.arange(256).reshape(1,1,16,16)
    patches = image.unfold(2,4,4).unfold(3,4,4).permute(0,2,3,1,4,5).reshape(1,16,16)
    assert torch.equal(model.unpatchify(patches),image)


def test_zero_initialization_and_required_context():
    model = tiny()
    x, a, ctx = torch.randn(2,2,16,16),torch.tensor([[.2],[.9]]),torch.randn(2,4,24)
    assert torch.equal(model(x,a,ctx),torch.zeros(2,1,16,16))
    with pytest.raises(ValueError,match='Prithvi'): model(x,a)


def test_time_and_context_affect_prediction_and_receive_gradients():
    model = tiny();activate(model)
    x, a = torch.randn(2,2,16,16),torch.tensor([[.2],[.9]])
    ctx = torch.randn(2,4,24,requires_grad=True)
    y = model(x,a,ctx)
    assert not torch.allclose(y,model(x,a,torch.zeros_like(ctx)))
    assert not torch.allclose(y,model(x,1-a,ctx))
    y.square().mean().backward()
    assert torch.isfinite(ctx.grad).all() and ctx.grad.abs().sum()>0
    for block in model.blocks:
        assert block.cross_attention.kv.weight.grad.abs().sum()>0


def test_checkpoint_forward_backward_equivalence():
    plain=tiny();activate(plain)
    checked=copy.deepcopy(plain);checked.gradient_checkpointing=True
    x,a,c=torch.randn(2,2,16,16),torch.rand(2,1),torch.randn(2,4,24)
    y1,y2=plain(x,a,c),checked(x,a,c)
    torch.testing.assert_close(y1,y2)
    y1.square().mean().backward();y2.square().mean().backward()
    for p,q in zip(plain.parameters(),checked.parameters()):
        torch.testing.assert_close(p.grad,q.grad)


@pytest.mark.parametrize('objective',['v','score'])
def test_diffusion_training_and_sampling(objective):
    class Encoder(torch.nn.Module):
        def forward(self,hls):return hls
    net=VScoreDiffusion(tiny(),Encoder(),objective)
    optimizer=torch.optim.Adam(net.denoise_fn.parameters(),lr=.001)
    batch={'HR':torch.rand(2,1,16,16),'SR':torch.rand(2,1,16,16),
           'mask':torch.ones(2,1,16,16).bool(),'HLS':torch.randn(2,4,24)}
    for _ in range(3):
        optimizer.zero_grad();loss=net(batch);loss.backward();optimizer.step()
        assert torch.isfinite(loss)
    grad=net.denoise_fn.blocks[0].cross_attention.kv.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    net.eval()
    out=net.sample(batch,5,torch.randn_like(batch['HR']))
    assert out.shape==batch['HR'].shape and torch.isfinite(out).all()
