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
"""Correctness and optional perf/stress for AMD intra-node all-reduce (rocSHMEM).

Analogous to ``test/nvidia/test_allreduce.py``, restricted to ``one_shot`` and
``two_shot`` (see ``kernels/amd/allreduce.py``).

Perf mode benchmarks Triton all-reduce against RCCL (``torch.distributed.all_reduce``
uses RCCL on ROCm; see project notes).
"""
import argparse
import itertools
import os
import random
import sys
from typing import Optional

import torch
import torch.distributed as dist
import triton

from triton_dist.kernels.allreduce import AllReduceMethod, to_allreduce_method
from triton_dist.kernels.amd.allreduce import all_reduce, create_allreduce_ctx
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (
    finalize_distributed,
    get_max_gpu_clock_rate_in_khz,
    initialize_distributed,
    sleep_async,
)

_AMD_METHODS = ("one_shot", "two_shot")

DATA_SIZES = [
    128,
    1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
]


def _pretty_format(nbytes):
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024}KB"
    if nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024}MB"
    return f"{nbytes / 1024 / 1024 / 1024}GB"


def _algo_hw_bw_gbps(nbytes: int, duration_ms: float, world_size: int, is_one_shot: bool):
    """Algorithm and hardware bandwidth (GB/s), matching ``run_perf`` conventions."""
    algo_bw = nbytes * 1e-9 / (duration_ms * 1e-3) * 2
    if world_size <= 1:
        hw_bw = algo_bw
    elif is_one_shot:
        hw_bw = algo_bw * world_size // 2
    else:
        hw_bw = algo_bw * (world_size - 1) / world_size
    return algo_bw, hw_bw


def _create_data(numel, dtype=torch.float32):
    inp = torch.rand((numel, ), dtype=dtype, device="cuda")
    if args.debug:
        inp = inp.fill_((RANK + 1) / 10)
    return inp


def torch_all_reduce(local_input: torch.Tensor, pg: torch.distributed.ProcessGroup):
    output = torch.clone(local_input)
    dist.all_reduce(output, group=pg)
    return output


def _randint_with_align(max_M, alignment: int):
    return random.randint(1, max_M // alignment) * alignment


def _random_straggler_option():
    rank = random.randint(0, WORLD_SIZE - 1)
    clock_hz = get_max_gpu_clock_rate_in_khz(0) * 1e3
    cycles = random.randint(0, int(clock_hz * 0.01))
    return (rank, cycles)


def stress_test(dtype: torch.dtype, args, method: AllReduceMethod):
    random.seed(args.seed)

    atol, rtol = {
        torch.bfloat16: (3e-2, 3e-2),
        torch.float16: (1e-2, 1e-2),
        torch.float32: (1e-3, 1e-3),
    }[dtype]

    align = args.alignment
    if method == AllReduceMethod.TwoShot:
        align = max(align, WORLD_SIZE)
    ctx = create_allreduce_ctx(args.max_nbytes, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE)

    def _all_reduce_with_output(x):
        out = torch.empty_like(x)
        all_reduce(x, method=method, ctx=ctx, output=out)
        return out

    for n in range(args.iters):
        tensor_inputs = [
            _create_data(
                _randint_with_align(args.max_nbytes // dtype.itemsize, align),
                dtype=dtype,
            ) for _ in range(args.verify_shapes)
        ]
        triton_out_list = [_all_reduce_with_output(x) for x in tensor_inputs]
        torch_out_list = [torch_all_reduce(x, pg=TP_GROUP) for x in tensor_inputs]

        try:
            for i, (triton_res, torch_res) in enumerate(zip(triton_out_list, torch_out_list)):
                assert_allclose(triton_res, torch_res, atol=atol, rtol=rtol, verbose=False)
        except Exception as e:
            print(f"RANK = {RANK}, {i}-th iteration failed with {e}", file=sys.stderr)
            raise e

        sleep_async(1000)
        for x in itertools.islice(itertools.cycle(tensor_inputs), args.verify_hang):
            straggler_opt = _random_straggler_option() if args.simulate_straggler else None
            all_reduce(x, method=method, ctx=ctx, straggler_option=straggler_opt)

        print(f"runs {n + 1} iterations done")
        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if TP_GROUP.rank() == 0:
        print(f"✅ AllReduce {method.name} Pass!")

    ctx.finalize()


def _is_one_shot(method: AllReduceMethod) -> bool:
    return method == AllReduceMethod.OneShot


def run_perf(dtype: torch.dtype, method: AllReduceMethod, warmup=5, iters=10):
    bytes_per_elem = dtype.itemsize
    available_ds = DATA_SIZES
    ctx = create_allreduce_ctx(available_ds[-1], RANK, WORLD_SIZE, LOCAL_WORLD_SIZE)
    is_one_shot = _is_one_shot(method)
    ratio_samples = []

    if RANK == 0:
        print(
            "Bandwidth vs RCCL: RCCL is measured with torch.distributed.all_reduce "
            "(RCCL on ROCm). Same warmup/iters and per-size message volume as Triton."
        )
        print(
            f"{'size':>8}  {'Triton algo':>12}  {'RCCL algo':>12}  {'Triton/RCCL':>12}  "
            f"{'Triton HW':>12}  {'RCCL HW':>12}  {'latency Triton':>14}  {'latency RCCL':>14}"
        )
        print(
            f"{'':>8}  {'GB/s':>12}  {'GB/s':>12}  {'(algo)':>12}  "
            f"{'GB/s':>12}  {'GB/s':>12}  {'us':>14}  {'us':>14}"
        )

    for nbytes in available_ds:
        num_elem = nbytes // bytes_per_elem
        if method == AllReduceMethod.TwoShot and num_elem % WORLD_SIZE != 0:
            continue
        local_input = _create_data(num_elem, dtype=dtype)
        rccl_buf = local_input.clone()

        def allreduce_op():
            all_reduce(local_input, method=method, ctx=ctx)

        sleep_async(100)
        _, triton_ms = perf_func(allreduce_op, warmup_iters=warmup, iters=iters)
        triton_algo, triton_hw = _algo_hw_bw_gbps(nbytes, triton_ms, WORLD_SIZE, is_one_shot)

        def rccl_allreduce_op():
            dist.all_reduce(rccl_buf, group=TP_GROUP)

        sleep_async(100)
        _, rccl_ms = perf_func(rccl_allreduce_op, warmup_iters=warmup, iters=iters)
        rccl_algo, rccl_hw = _algo_hw_bw_gbps(nbytes, rccl_ms, WORLD_SIZE, is_one_shot=False)

        if rccl_algo > 0:
            ratio_samples.append(triton_algo / rccl_algo)

        if RANK == 0:
            ratio_str = f"{triton_algo / rccl_algo:0.3f}" if rccl_algo > 0 else "n/a"
            print(
                f"{_pretty_format(nbytes):>8}  {triton_algo:12.2f}  {rccl_algo:12.2f}  {ratio_str:>12}  "
                f"{triton_hw:12.2f}  {rccl_hw:12.2f}  {triton_ms * 1000:14.2f}  {rccl_ms * 1000:14.2f}"
            )

    ctx.finalize()

    if RANK == 0 and ratio_samples:
        mean_ratio = sum(ratio_samples) / len(ratio_samples)
        if mean_ratio > 1.0:
            print(
                f"\nSummary: mean algorithm-bandwidth ratio Triton/RCCL = {mean_ratio:.3f} "
                f"over {len(ratio_samples)} sizes (Triton higher by ~{mean_ratio:.2f}x on average)."
            )
        elif mean_ratio < 1.0:
            inv = 1.0 / mean_ratio
            print(
                f"\nSummary: mean algorithm-bandwidth ratio Triton/RCCL = {mean_ratio:.3f} "
                f"over {len(ratio_samples)} sizes (RCCL higher by ~{inv:.2f}x on average)."
            )
        else:
            print(f"\nSummary: mean Triton/RCCL algorithm bandwidth ≈ 1.0 over {len(ratio_samples)} sizes.")


def _triton_warmup():
    triton.compiler.compiler.triton_key()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_nbytes", type=int, default=1024 * 4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=25)
    parser.add_argument("--verify_shapes", type=int, default=200)
    parser.add_argument("--verify_hang", type=int, default=100)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--method", type=str, default="two_shot", choices=_AMD_METHODS)
    parser.add_argument("--simulate_straggler", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--stress", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    DTYPE = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=DTYPE)

    triton.set_allocator(alloc_fn)

    method = to_allreduce_method(args.method)

    if args.stress:
        stress_test(DTYPE, args, method=method)
    else:
        _triton_warmup()
        _run_id = os.environ.get("TORCHELASTIC_RUN_ID", "0")
        with group_profile(f"all_reduce_amd_{_run_id}", args.profile, group=TP_GROUP):
            run_perf(DTYPE, method, warmup=args.warmup_iters, iters=args.iters)

    finalize_distributed()
