import MultiScaleDeformableAttention as MSDA
from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
shapes = torch.tensor([[4,4]], device='cuda', dtype=torch.long)
start = torch.tensor([0], device='cuda', dtype=torch.long)
v = torch.randn(1, 16, 2, 8, device='cuda', requires_grad=True)
loc = torch.rand(1, 3, 2, 1, 4, 2, device='cuda', requires_grad=True)
attn = torch.softmax(torch.randn(1, 3, 2, 1, 4, device='cuda'), dim=-1).detach().requires_grad_(True)
out = MSDeformAttnFunction.apply(v, shapes, start, loc, attn, 1)
assert out.shape == (1,3,16), out.shape
assert torch.isfinite(out).all()
out.square().mean().backward()
assert v.grad is not None and torch.isfinite(v.grad).all()
print('✅ Operator CUDA forward + backward sukses:',tuple(out.shape))
