import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here

    pid = tl.program_id(axis=0)
    offs_col = tl.arange(0, BLOCK_SIZE)
    mask = offs_col< n_cols

    x_ptrs = x_ptr + pid * x_row_stride + offs_col

    
    x = tl.load(x_ptrs, mask = mask, other = -float('inf'))
    m = tl.max(x, axis=0)
    sum = tl.sum(tl.exp(x-m),axis =0)
    lse = m + tl.log(sum)

    tl.store(out_ptr+pid, lse)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )