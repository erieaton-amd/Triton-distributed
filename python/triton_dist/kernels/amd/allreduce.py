################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Intra-node all-reduce via rocSHMEM (put + signal + ring reduce).

Mirrors the one-shot and two-shot push algorithms in ``kernels/nvidia/allreduce.py``,
adapted for HIP using ``libshmem_device`` (rocSHMEM): ``putmem_signal_nbi_wave``,
``fence``, ``signal_wait_until`` with ``ROCSHMEM_*`` constants, and ``barrier_all``.

Multimem, TMA, and double-tree variants are not supported on this path.
"""
import dataclasses
from typing import Optional

import pyrocshmem
import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.hip.librocshmem_device import set_rocshmem_ctx
from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.kernels.amd.common_ops import barrier_on_this_grid
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import __syncthreads, tid
from hip import hip

from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (
    NVSHMEM_SIGNAL_DTYPE,
    get_device_property,
    launch_cooperative_grid_options,
    rocshmem_barrier_all_on_stream,
)

MAX_DOUBLE_TREE_BLOCKS = 1024


def workspace_bytes_per_in_byte(world_size, method: AllReduceMethod) -> int:
    if method in [AllReduceMethod.OneShot, AllReduceMethod.OneShot_TMA]:
        return world_size
    if method in [AllReduceMethod.TwoShot]:
        return 2
    if method in [
            AllReduceMethod.OneShot_Multimem,
            AllReduceMethod.TwoShot_Multimem,
    ]:
        raise NotImplementedError(f"AllReduce method {method} is not implemented for AMD/rocSHMEM")
    raise ValueError(f"Unknown allreduce method {method}")


def get_max_chunk_nbytes(workspace_nbytes, world_size, method: AllReduceMethod) -> int:
    return workspace_nbytes // workspace_bytes_per_in_byte(world_size, method)


@dataclasses.dataclass
class AllReduceContext:
    workspace_nbytes: int
    rank: int
    world_size: int
    local_world_size: int
    symm_scatter_buf: torch.Tensor
    symm_signal: torch.Tensor
    phase: int = 0
    grid_barrier: torch.Tensor = dataclasses.field(init=False)
    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        self.nnodes = self.world_size // self.local_world_size
        self.grid_barrier = torch.zeros((1024, ), dtype=torch.int32, device="cuda")

    def finalize(self):
        torch.cuda.synchronize()
        del self.symm_scatter_buf
        del self.symm_signal


def create_allreduce_ctx(workspace_nbytes, rank, world_size, local_world_size) -> AllReduceContext:
    symm_scatter_buf = pyrocshmem.rocshmem_create_tensor((workspace_nbytes, ), torch.int8)
    symm_signal = pyrocshmem.rocshmem_create_tensor((MAX_DOUBLE_TREE_BLOCKS * world_size, ), NVSHMEM_SIGNAL_DTYPE)
    symm_signal.fill_(0)
    rocshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    return AllReduceContext(
        workspace_nbytes=workspace_nbytes,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        symm_scatter_buf=symm_scatter_buf,
        symm_signal=symm_signal,
    )


def _run_straggler(ctx, straggler_option):
    if straggler_option:
        rank, cycles = straggler_option
        if rank == ctx.rank:
            torch.cuda._sleep(cycles)


@triton_dist.jit
def kernel_ring_reduce_non_tma(
    in_ptr,
    out_ptr,
    elems_per_rank,
    begin_idx,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    num_blocks = tl.cdiv(elems_per_rank, BLOCK_SIZE)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    for n in range(pid, num_blocks, npid):
        segment = (begin_idx + 1) % NUM_SPLITS
        c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        mask = c_offs < elems_per_rank * (segment + 1)
        accum = tl.load(in_ptr + c_offs, mask=mask)
        for i in range(1, NUM_SPLITS):
            segment = (i + begin_idx + 1) % NUM_SPLITS
            c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
            data = tl.load(in_ptr + c_offs, mask=mask)
            accum += data
        out_offs = BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + out_offs, accum, mask=mask)


@triton_dist.jit
def copy_continuous_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)
    for n in range(pid, n_blocks, num_pid):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        val = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, val, mask=mask)


@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_one_shot_push_intra_node_kernel(
    ctx,
    input_ptr,
    output_ptr,
    symm_signal_ptr,
    symm_buffer_ptr,
    grid_barrier_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    symm_buffer_ptr = tl.cast(symm_buffer_ptr, input_ptr.dtype)
    nbytes = tl.cast(n_elements * elem_size, tl.uint64)
    sig_one = tl.cast(1, tl.uint64)

    if pid == 0:
        offs = tl.arange(0, world_size)
        tl.store(symm_signal_ptr + offs, tl.zeros([world_size], dtype=tl.int64))
        if thread_idx == 0:
            libshmem_device.barrier_all()
    __syncthreads()

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_wave(
            symm_buffer_ptr + n_elements * rank,
            input_ptr,
            nbytes,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    libshmem_device.fence()

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()

    kernel_ring_reduce_non_tma(
        symm_buffer_ptr,
        output_ptr,
        n_elements,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_two_shot_push_intra_node_kernel(
    ctx,
    input_ptr,
    symm_out_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    out_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    elem_per_rank = tl.cdiv(n_elements, world_size)
    symm_out_ptr = tl.cast(symm_out_ptr, input_ptr.dtype)
    symm_recv_ptr = symm_out_ptr + n_elements
    nbytes_shard = tl.cast(elem_per_rank * elem_size, tl.uint64)
    sig_one = tl.cast(1, tl.uint64)

    if pid == 0:
        offs = tl.arange(0, world_size * 2)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_wg()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    if pid < world_size:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_wg(
            symm_recv_ptr + rank * elem_per_rank,
            input_ptr + peer * elem_per_rank,
            nbytes_shard,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()
    libshmem_device.fence()

    kernel_ring_reduce_non_tma(
        symm_recv_ptr,
        symm_out_ptr + elem_per_rank * rank,
        elem_per_rank,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    symm_signal_ptr += world_size
    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_wg(
            symm_out_ptr + rank * elem_per_rank,
            symm_out_ptr + rank * elem_per_rank,
            nbytes_shard,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    libshmem_device.fence()

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()
    libshmem_device.fence()

    copy_continuous_kernel(symm_out_ptr, out_ptr, n_elements, BLOCK_SIZE)

@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_two_shot_push_intra_node_kernel_v2(
    ctx,
    input_ptr,
    symm_out_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    out_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    elem_per_rank = tl.cdiv(n_elements, world_size)
    symm_out_ptr = tl.cast(symm_out_ptr, input_ptr.dtype)
    symm_recv_ptr = symm_out_ptr + n_elements
    nbytes_shard = tl.cast(elem_per_rank * elem_size, tl.uint64)
    sig_one = tl.cast(1, tl.uint64)
    
    num_blocks = tl.cdiv(elem_per_rank, BLOCK_SIZE)
    total_blocks = num_blocks * world_size

    if pid == 0:
        offs = tl.arange(0, world_size * 1024)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_wg()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    for job in range(pid, total_blocks, num_sms):
        peer = job % world_size
        block = job // world_size

        src_ptr = input_ptr + peer * elem_per_rank;
        peer_ptr = dl.symm_at(symm_recv_ptr + rank * elem_per_rank, peer)
            
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elem_per_rank
        src_ptrs = src_ptr + offsets
        peer_ptrs = peer_ptr + offsets
        tl.store(peer_ptrs, tl.load(src_ptrs, mask=mask), mask=mask)
        if thread_idx == 0:
            peer_signal = dl.symm_at(symm_signal_ptr + block * world_size + rank, peer)
            tl.atomic_add(peer_signal, 1, sem="release", scope="sys")

    if thread_idx < world_size:
        for block in range(num_blocks):
            offset = block * world_size + thread_idx
            libshmem_device.signal_wait_until(
                symm_signal_ptr + offset,
                libshmem_device.ROCSHMEM_CMP_EQ,
                sig_one,
            )
    
    __syncthreads()
    libshmem_device.fence()

    kernel_ring_reduce_non_tma(
        symm_recv_ptr,
        symm_out_ptr + elem_per_rank * rank,
        elem_per_rank,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    symm_signal_ptr += total_blocks
    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_wg(
            symm_out_ptr + rank * elem_per_rank,
            symm_out_ptr + rank * elem_per_rank,
            nbytes_shard,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    libshmem_device.fence()

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()
    libshmem_device.fence()

    copy_continuous_kernel(symm_out_ptr, out_ptr, n_elements, BLOCK_SIZE)


@triton_dist.jit(do_not_specialize=["group_rank"])
def persistent_all_reduce_two_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Reduce assigned tiles for a rank and broadcast the result to all peers.
    Single kernel: unmasked fast path for full tiles, masked slow path for tails.
    """
    pid = tl.program_id(0)
    COMM_SMS = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    # Persistent traversal
    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        #base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            base_rank_ptr = dl.symm_at(input_ptr, start_rank_global) + input_offset
            acc = tl.load(base_rank_ptr).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                base_rank_ptr = dl.symm_at(input_ptr, remote_rank) + input_offset
                acc += tl.load(base_rank_ptr).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    out_rank_ptr = dl.symm_at(output_ptr, remote_rank) + output_offset
                    tl.store(out_rank_ptr, reduced)

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            base_rank_ptr = dl.symm_at(input_ptr, start_rank_global) + input_offset
            acc = tl.load(base_rank_ptr, mask=mask)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                base_rank_ptr = dl.symm_at(input_ptr, remote_rank) + input_offset
                acc += tl.load(base_rank_ptr, mask=mask).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    out_rank_ptr = dl.symm_at(output_ptr, remote_rank) + output_offset
                    tl.store(out_rank_ptr, reduced, mask=mask)


def allreduce_one_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 16,
):
    assert x.is_cuda and x.is_contiguous()
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.shape == output.shape
    assert x.nbytes <= ctx.workspace_nbytes // ctx.world_size

    block_size = num_warps * 64 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = min(max(get_device_property().multi_processor_count - 4, 1), num_tiles)
    dev_ctx = pyrocshmem.rocshmem_get_device_ctx()
    allreduce_one_shot_push_intra_node_kernel[(num_tiles, )](
        dev_ctx,
        x,
        output,
        ctx.symm_signal,
        ctx.symm_scatter_buf,
        ctx.grid_barrier,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        use_cooperative=False,
        **launch_cooperative_grid_options(),
    )
    return output


def allreduce_two_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 16,
):
    assert x.numel() % ctx.world_size == 0, "two_shot allreduce requires numel divisible by world_size"
    assert x.is_cuda and x.is_contiguous()
    if output is not None:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.nbytes == output.nbytes
    else:
        output = torch.empty_like(x)
    assert x.nbytes <= ctx.workspace_nbytes // 2

    block_size = num_warps * 64 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = max(ctx.world_size, min(get_device_property().multi_processor_count, num_tiles))
    dev_ctx = pyrocshmem.rocshmem_get_device_ctx()

    # allreduce_two_shot_push_intra_node_kernel[(num_tiles, )](
    #     dev_ctx,
    #     x,
    #     ctx.symm_scatter_buf,
    #     ctx.symm_signal,
    #     ctx.grid_barrier,
    #     output,
    #     ctx.rank,
    #     ctx.world_size,
    #     num_elem,
    #     BLOCK_SIZE=block_size,
    #     num_warps=num_warps,
    #     use_cooperative=False,
    #     **launch_cooperative_grid_options(),
    # )

    # Copy from non-symmetric memory to symmetric memory. The iris-based kernels don't do this for you.
    hip.hipMemcpy(
        ctx.symm_scatter_buf[x.nbytes:x.nbytes * 2],
        x.data_ptr(),
        x.nbytes,
        hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    )
    pyrocshmem.rocshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    extra_dim = 16
    if num_elem % extra_dim != 0:
        raise RuntimeError("Bad dimension")
    reshape_2d = (num_elem // extra_dim, extra_dim)
    input_buf = ctx.symm_scatter_buf[x.nbytes:x.nbytes * 2].view(dtype=x.dtype).view(reshape_2d)
    output_buf = ctx.symm_scatter_buf[:x.nbytes].view(dtype=x.dtype).view(reshape_2d)
    persistent_all_reduce_two_shot[(num_tiles, )](
        input_buf,
        output_buf,
        input_buf.shape[0],
        input_buf.shape[1],
        input_buf.stride(0),
        input_buf.stride(1),
        output_buf.stride(0),
        output_buf.stride(1),
        ctx.rank,
        ctx.rank,
        ctx.world_size,
        0,
        1,
        block_size // extra_dim,
        extra_dim,
        4,
        0
    )
    pyrocshmem.rocshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    hip.hipMemcpy(
        output.data_ptr(), 
        ctx.symm_scatter_buf[:x.nbytes],
        x.nbytes,
        hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    )

    return output


def get_auto_allreduce_method_amd(nbytes: int, world_size: int) -> AllReduceMethod:
    """Prefer two-shot when symmetric memory for one-shot would dominate."""
    if nbytes * world_size <= 64 * 1024 * 1024:
        return AllReduceMethod.OneShot
    return AllReduceMethod.TwoShot


def all_reduce(
    x: torch.Tensor,
    method: Optional[AllReduceMethod],
    ctx: AllReduceContext,
    output: Optional[torch.Tensor] = None,
    max_sm: int = -1,
    straggler_option=None,
):
    method = method or get_auto_allreduce_method_amd(x.nbytes, ctx.world_size)
    if method not in (AllReduceMethod.OneShot, AllReduceMethod.TwoShot):
        raise NotImplementedError(
            f"AMD all_reduce supports only OneShot and TwoShot; got {method}. "
            "Use kernels/nvidia/allreduce.py for multimem/TMA/double-tree.")

    op_handle = {
        AllReduceMethod.OneShot: allreduce_one_shot_push_intra_node,
        AllReduceMethod.TwoShot: allreduce_two_shot_push_intra_node,
    }[method]

    nbytes_per_chunk = ctx.workspace_nbytes // workspace_bytes_per_in_byte(ctx.world_size, method)
    nchunks = triton.cdiv(x.nbytes, nbytes_per_chunk)
    elems_per_chunk = nbytes_per_chunk // x.itemsize

    if nchunks == 1:
        return op_handle(
            ctx=ctx,
            x=x,
            output=output,
            max_sm=max_sm,
            num_warps=16,
            straggler_option=straggler_option,
        )
    if output is None:
        output = torch.empty_like(x)
    for n in range(nchunks):
        op_handle(
            ctx=ctx,
            x=x.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            output=output.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            max_sm=max_sm,
            num_warps=16,
            straggler_option=straggler_option,
        )
    return output
