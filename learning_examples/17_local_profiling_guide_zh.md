# 本地多层级 Profiling 学习笔记

## 1. 本阶段目标

当前 PP4 实验已经在同构设备上验证：非均匀全层分配可以改善首尾 stage
负载，半层边界可以在大序列场景继续细化计算平衡。12 层模型不代表真实大模型，
但它验证了统一规划、细粒度构建和通信边界能够工作。

下一阶段先不追求新的加速结论，而是建立一套可以迁移到多节点 Megatron 的
profiling 方法：

1. 正确采集，不让 profiler 本身主导结果。
2. 从训练 step 一直定位到 CUDA kernel 和硬件限制。
3. 区分计算瓶颈、显存瓶颈、通信等待和 CPU launch 间隙。
4. 每个结论都能回到一段 timeline、一个 operator 或一组 counter。

本地环境：RTX 4060 Laptop 8 GB、PyTorch 2.12.1+cu126、Nsight Systems
2024.2.3、Nsight Compute 2024.2.1。

## 2. 分析层级

```text
训练迭代
  -> forward / backward / optimizer
    -> Transformer layer / attention / MLP
      -> PyTorch operator
        -> CUDA kernel
          -> SM、Tensor Core、L2、DRAM、warp stall
```

不要一开始就用 NCU 检查所有 kernel。先用 wall time 确认现象，再用 Nsight
Systems 找到时间段和热点 kernel，然后用 PyTorch Profiler 建立 operator 到
kernel 的映射，最后只用 NCU 检查少数关键 kernel。

## 3. 基线：无 profiler

```powershell
F:\PycharmProjects\llm\.venv\Scripts\python.exe `
  learning_examples\profile_transformer_local.py
```

先记录 median、P95 和 peak allocated。所有 profiler 结果都与这次无采集运行
比较。如果开启 profiler 后 step time 改变很大，只把采集结果用于定位，不把它
当作真实性能数据。

## 4. PyTorch Profiler：operator 层

```powershell
F:\PycharmProjects\llm\.venv\Scripts\python.exe `
  learning_examples\profile_transformer_local.py `
  --profiler torch
```

输出包括 CUDA self time 排名前 20 的 operator 和 Chrome trace JSON。

重点字段：

- `Self CUDA`：operator 自己发起的 CUDA 工作，不含子 operator。
- `CUDA total`：包含子调用的 CUDA 总时间。
- `# of Calls`：小 kernel 是否被过度重复调用。
- Input Shapes：是否存在不理想的小 GEMM 或异常 shape。
- Self CUDA Memory：operator 自身分配/释放的 tensor 显存。
- CPU Self Time：Python、dispatcher 或 CPU 工作是否成为 launch 瓶颈。

`record_shapes`、`profile_memory` 和 `with_stack` 都有额外开销；默认脚本不开
`with_stack`。只有需要回溯源码时才加 `--with-stack`。

## 5. Nsight Systems：系统和 timeline 层

### 5.1 采集

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.2.3\target-windows-x64\nsys.exe' `
  profile --trace=cuda,nvtx --sample=none `
  --output=profile_outputs\local_transformer\nsys_transformer `
  F:\PycharmProjects\llm\.venv\Scripts\python.exe `
  learning_examples\profile_transformer_local.py `
  --steps 10 --warmup 4
```

这里使用 Windows 版 Nsight Systems 2024.2.3，它不接受 Linux 版常用的
`osrt` trace。迁移到远程 Linux 时再加入 `osrt`，用于观察线程调度、锁和系统调用。

打开 GUI：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.2.3\host-windows-x64\nsys-ui.exe' `
  profile_outputs\local_transformer\nsys_transformer.nsys-rep
```

### 5.2 先看什么

1. NVTX 的 iteration、forward、backward、optimizer 范围是否完整。
2. GPU stream 上是否有大段空白；空白前 CPU 在做什么。
3. CUDA API launch 与 GPU kernel 是否连续，CPU 是否来不及发射。
4. `cudaDeviceSynchronize`、`cudaStreamSynchronize` 等同步调用是否过多。
5. H2D/D2H memcpy 是否落在关键路径，是否和计算重叠。
6. kernel 是否形成少数长 kernel，还是大量很短的 kernel。
7. 后续 Megatron 中 NCCL 是否与计算重叠，还是形成串行等待。

Nsight Systems 回答的是“时间花在哪里、谁在等谁”，不能仅凭 timeline 判断
某个 kernel 是 compute bound 还是 memory bound。

## 6. Nsight Compute：kernel 和硬件层

先用 Nsys 找到热点 kernel，再限制 launch 数量。不要对整个训练使用 `--set full`。

第一次只验证 kernel 选择和 launch 信息：

```powershell
New-Item -ItemType Directory -Force profile_outputs\local_transformer | Out-Null
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.1\target\windows-desktop-win7-x64\ncu.exe' `
  --section LaunchStats --launch-skip 20 --launch-count 1 `
  --export profile_outputs\local_transformer\ncu_transformer `
  F:\PycharmProjects\llm\.venv\Scripts\python.exe `
  learning_examples\profile_transformer_local.py `
  --steps 8 --warmup 3
```

在 Nsys 中确定热点 kernel 名称后，再用 `--kernel-name` 限定一个 kernel，并逐步
增加 `SpeedOfLight`、`Occupancy`、`SchedulerStats` 和 `WarpStateStats` section。
Windows WDDM 下 NCU 注入和 replay 可能非常慢；本地首次 `basic` 单 kernel 采集
超过 120 秒仍未完成，因此不要直接复制 Linux 服务器上的 full-set 采集策略。

打开 GUI：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.1\host\windows-desktop-win7-x64\ncu-ui.exe' `
  profile_outputs\local_transformer\ncu_transformer.ncu-rep
```

重点 section 和指标：

- Speed of Light：Compute 与 Memory throughput 谁更接近上限。
- Roofline：算术强度与实际 FLOP/s，判断 compute bound 或 memory bound。
- Occupancy：理论/实际 occupancy；高 occupancy 不等于高性能。
- Launch Statistics：grid、block、register/thread、shared memory。
- Scheduler Statistics：active、eligible、issued warp 是否充足。
- Warp State：只有 scheduler 发射不足时，再分析 memory dependency、execution
  dependency、barrier 等 stall。
- Memory Workload：DRAM、L2、L1/TEX 吞吐和 cache hit rate。
- Tensor Core：矩阵 kernel 是否真正走 Tensor Core 路径。

NCU 经常需要多次 replay kernel，采集本身会严重变慢。它用于解释 kernel，
不用于测量端到端训练时间。

## 7. 从现象到结论的顺序

### GPU timeline 有空洞

先检查 CPU launch、Python/data loader、同步调用，再检查 kernel 本身。不要直接
把空洞归因于 GPU 算力不足。

### GPU 连续忙但 step 仍慢

从 Nsys 的 kernel summary 找热点，再用 NCU 判断 compute、memory、occupancy 或
stall。连续忙只说明有工作，不说明工作高效。

### 显存高但算力低

检查 activation 生命周期、并发 microbatch、临时 workspace 和 allocator；参数量
不是 PP stage 显存的唯一来源。

### 通信占比高

分别统计 NCCL duration、通信等待、compute/communication overlap 和消息大小。
通信 kernel 时间长不一定表示通信是关键路径；只有阻塞后续计算才构成 step 瓶颈。

## 8. 迁移到 Megatron PP

本地练习稳定后，在 Megatron 中加入或复用少量 NVTX range：

- iteration / train_step；
- warmup / steady 1F1B / cooldown；
- microbatch；
- forward / backward；
- attention / MLP；
- send / recv；
- optimizer。

远程第一次只 profile 两个方案：均匀 `[3,3,3,3]` 和当前候选
half `[4,8,9,3]`。每个 rank 单独保留 report，记录 commit、GPU、驱动、CUDA、
NCCL、模型参数和完整命令。先验证：

1. 首尾 stage 的额外计算在哪里；
2. `[4,8,9,3]` 是否缩短了关键 stage，而不是只转移等待；
3. Attention/MLP 边界通信是否落在关键路径；
4. 不同 rank 的 activation 峰值为何不同；
5. 1F1B 稳态中的计算通信重叠是否改善。

## 9. 常见误区

- 只看 GPU utilization 百分比，不看 timeline 和关键路径。
- 把 occupancy 当作性能目标，而不是延迟隐藏条件。
- 对整个训练运行 NCU full set，产生巨大 replay 开销。
- 比较不同 shape、不同 backend、不同频率状态下的 kernel counter。
- 用 profiler 下的 wall time 代替无 profiler 基线。
- 在一次运行中同时开启所有 tracing、stack、memory 和硬件 counter。
- 只看 NCCL 总时间，不判断通信是否阻塞计算。

## 10. 官方资料

- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/
- Nsight Systems Analysis Guide: https://docs.nvidia.com/nsight-systems/AnalysisGuide/
- Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch Profiler Recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- PyTorch NVTX: https://docs.pytorch.org/docs/stable/cuda.html#nvidia-tools-extension-nvtx

## 11. 本地首次采集记录

使用 `batch=2, seq=256, hidden=512, layers=2`：

- 无 profiler：median 约 4.68 ms，peak allocated 约 143 MiB。
- PyTorch Profiler：median 约 13.68 ms，说明 profiler wall time 不能当基线。
- Nsys：iteration 0 约 250 ms，稳定 iteration 约 6 到 9 ms。
- Nsys kernel summary：小 workload 中 AdamW foreach kernel 的累计占比高于任一
  GEMM；这不等于大模型也由 optimizer 主导。
- Attention 使用 `fmha_cutlass` kernel，FP16 GEMM 使用 Ampere Tensor Core kernel。
- NCU basic 单 kernel 在 Windows WDDM 下超过 120 秒，已终止且未作为有效采集。

PyTorch Profiler 表格中的父 range 与子 operator 会重复包含同一段 CUDA 时间，
因此百分比可能超过 100%。分析时优先比较同一层级的 self time，不要把嵌套范围
直接相加。
