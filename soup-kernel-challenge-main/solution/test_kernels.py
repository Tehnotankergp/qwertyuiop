"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def _ref_rmsnorm(x, w, eps):
    # exactly Qwen2RMSNorm.forward of the base
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_rmsnorm_matches_pytorch(w_dtype, shape):
    from solution.rmsnorm import rms_norm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    w = (torch.randn(shape[-1], device="cuda") * 0.5 + 1).to(w_dtype)
    grad = torch.randn(shape, device="cuda").to(torch.promote_types(torch.float16, w_dtype))

    ref = _ref_rmsnorm(x, w, 1e-6)
    ref.backward(grad)
    ref_dx = x.grad.float()
    x.grad = None

    out = rms_norm(x, w, 1e-6)
    out.backward(grad)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=5e-3, atol=5e-3)
