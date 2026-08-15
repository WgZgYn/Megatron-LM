#!/usr/bin/env python3
"""Small CUDA Transformer workload for PyTorch, Nsight Systems, and Nsight Compute."""

import argparse
import contextlib
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ln1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        residual = hidden_states
        mixed = self.qkv(self.ln1(hidden_states))
        query, key, value = mixed.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            split_heads(query), split_heads(key), split_heads(value), is_causal=True
        )
        context = context.transpose(1, 2).contiguous().view(batch, sequence, hidden)
        hidden_states = residual + self.proj(context)
        residual = hidden_states
        hidden_states = self.fc2(F.gelu(self.fc1(self.ln2(hidden_states))))
        return residual + hidden_states


class TinyTransformer(nn.Module):
    def __init__(self, layers, hidden_size, num_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_norm(hidden_states)


def nvtx_range(name, enabled):
    return torch.cuda.nvtx.range(name) if enabled else contextlib.nullcontext()


def build_profiler(args, output_dir):
    if args.profiler != 'torch':
        return None

    def save_trace(profiler):
        trace_path = output_dir / f'pytorch_trace_step_{profiler.step_num}.json'
        profiler.export_chrome_trace(str(trace_path))
        print(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=20))
        print(f'PyTorch trace: {trace_path}')

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=args.warmup, warmup=1, active=args.active, repeat=1
        ),
        on_trace_ready=save_trace,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiler', choices=('none', 'torch'), default='none')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--active', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--with-stack', action='store_true')
    parser.add_argument('--output-dir', default='profile_outputs/local_transformer')
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this profiling exercise')
    minimum_steps = args.warmup + 1 + args.active
    if args.profiler == 'torch' and args.steps < minimum_steps:
        raise ValueError(f'--steps must be at least {minimum_steps} for the profiler schedule')

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    model = TinyTransformer(args.num_layers, args.hidden_size, args.num_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    scaler = torch.amp.GradScaler('cuda')
    inputs = torch.randn(
        args.batch_size, args.seq_length, args.hidden_size, device=device, dtype=torch.float16
    )
    model.train()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    profiler = build_profiler(args, output_dir)
    profiler_context = profiler if profiler is not None else contextlib.nullcontext()

    torch.cuda.reset_peak_memory_stats()
    with profiler_context:
        for step in range(args.steps):
            start_events[step].record()
            with nvtx_range(f'iteration_{step}', True):
                optimizer.zero_grad(set_to_none=True)
                with nvtx_range('forward', True):
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        output = model(inputs)
                        loss = output.float().square().mean()
                with nvtx_range('backward', True):
                    scaler.scale(loss).backward()
                with nvtx_range('optimizer', True):
                    scaler.step(optimizer)
                    scaler.update()
            end_events[step].record()
            if profiler is not None:
                profiler.step()

    torch.cuda.synchronize()
    measured = [
        start_events[index].elapsed_time(end_events[index])
        for index in range(args.warmup, args.steps)
    ]
    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'steps: {args.steps}, measured: {len(measured)}')
    print(f'median step: {statistics.median(measured):.3f} ms')
    print(f'p95 step: {sorted(measured)[max(0, int(0.95 * len(measured)) - 1)]:.3f} ms')
    print(f'peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
