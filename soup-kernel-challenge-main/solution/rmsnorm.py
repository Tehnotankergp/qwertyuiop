"""Own kernel: fused RMSNorm (forward + backward) for Qwen2RMSNorm.

Base Qwen2RMSNorm does: x.float() -> pow -> mean -> rsqrt -> mul -> .to(input_dtype) -> weight * ...
That is ~6 small PyTorch ops, each reading/writing the whole tensor (and fp32 copies). Here one
launch reads x once, writes y once and saves only one fp32 number per row (rstd) for backward.

Numerics follow the base exactly: the normalised value is rounded to the input dtype *before*
multiplying by the weight, and the output dtype is the promoted dtype of (weight, x), so with an
fp32 weight (prepare_model_for_kbit_training) the output is fp32 like in the base.
Idea source: the standard fused RMSNorm design (one row per program, fp32 accumulation), as in the
Triton tutorials / common practice; the code is written from scratch for this task.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # base: hidden_states.to(float32) * rsqrt(...) then .to(input_dtype), then weight * that
    n = (xf * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * N + cols, (w * n).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rms_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    n = x * rstd
    dn = dy * w
    c = tl.sum(dn * n, axis=0) / N
    dx = rstd * (dn - n * c)
    tl.store(dx_ptr + row * N + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        n_cols = shape[-1]
        x2 = x.reshape(-1, n_cols).contiguous()
        weight = weight.contiguous()
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty(x2.shape, device=x.device, dtype=out_dtype)
        rstd = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n_cols)
        warps = 4 if block <= 1024 else 8
        rms_fwd_kernel[(x2.shape[0],)](x2, weight, y, rstd, n_cols, eps, BLOCK_N=block,
                                       num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block, ctx.warps = shape, block, warps
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n_cols = x2.shape[1]
        dy2 = dy.reshape(-1, n_cols).contiguous()
        dx = torch.empty_like(x2)
        rms_bwd_kernel[(x2.shape[0],)](dy2, x2, weight, rstd, dx, n_cols, BLOCK_N=ctx.block,
                                       num_warps=ctx.warps)
        dw = None
        if ctx.needs_input_grad[1]:  # frozen in this task; kept for correctness
            n = x2.float() * rstd[:, None]
            dw = (dy2.float() * n).sum(0).to(weight.dtype)
        return dx.reshape(ctx.shape), dw, None


def rms_norm(x, weight, eps):
    return RMSNormFn.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel (class patch, safe to call early)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = getattr(modeling_qwen2, "Qwen2RMSNorm", None)
    if cls is None:
        return
    original = cls.forward

    def forward(self, hidden_states):
        if not (isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda):
            return original(self, hidden_states)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    cls.forward = forward
