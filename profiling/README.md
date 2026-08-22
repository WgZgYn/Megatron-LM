# Profiling harness — DP / TP / PP 多卡通信分析

自包含的经典 transformer 分布式训练 profiling 工程。目标、口径与验证结果见
[`PLAN.md`](PLAN.md)。

## 文件

```
profiling/
├── model.py              # mini GPT transformer（TP-native，tp=1 即普通模型）
├── comm.py               # CommTimer：collective 计时 + 带宽/时延记录
├── run.py                # 主入口：dist init + 按 mode 建模型 + profiler + comm 记录
├── analyze.py            # 汇总 comm/steps → 表格 + CSV + 对比图
├── parallelism/
│   ├── dp.py             # DDP + 计时 comm hook
│   ├── tp.py             # Column/RowParallelLinear（含反向 all-reduce 的 autograd）
│   └── pp.py             # Gpipe pipeline stage + 阻塞 send/recv 调度
├── launchers/
│   ├── run_remote.sh     # 远程 torchrun（Linux + NCCL）
│   ├── nsys_wrap.sh      # Nsight Systems 包装
│   └── local_launch.py   # Windows 本地多进程启动器（GLOO CPU 冒烟）
├── PLAN.md               # 目标、方案、验证矩阵、分析口径
└── outputs/              # 运行产物（gitignore）
```

## 快速开始

### 远程（Linux + 多卡 NCCL）

```bash
# 三种策略各跑一次（同模型配置，便于对比）
bash launchers/run_remote.sh dp 4 cmp
bash launchers/run_remote.sh tp 4 cmp
bash launchers/run_remote.sh pp 4 cmp

# 分析
python analyze.py                       # 汇总表 + outputs/summary_all.csv + comm_comparison.png
tensorboard --logdir outputs/cmp_pp_p4/trace   # 时间轴（通信时机/overlap）

# （可选）Nsight Systems 交叉验证
bash launchers/nsys_wrap.sh pp 4 nsys_pp
```

`run.py` 关键参数：`--mode {dp,tp,pp}`、`--steps`、`--warmup`、`--hidden/--layers/--heads/...`、
`--global-batch`、`--num-microbatches`（PP）、`--dtype {fp32,bf16,fp16}`、
`--profile/--no-profile`、`--measure-comm/--no-measure-comm`、`--tag`。

### 混合精度 / Tensor Core

`--dtype` 用 `torch.autocast` 把前向的 matmul 切到 tensor core：

- **V100**（只有 FP16 tensor core）：用 `--dtype fp16`（带 GradScaler）。
- **A100/H100**（有 BF16 tensor core）：用 `--dtype bf16`（无需 GradScaler）。
- 默认 `fp32` = 纯 CUDA core，`volta_sgemm_*` kernel，无 tensor core。

`fp16/bf16` 会大幅缩短 compute，从而把**通信暴露成瓶颈**——这正是研究通信要看的真实场景
（V100-PCIe 下 FP32 compute 太慢会把 comm 盖住）。

### 通信时间从哪来

- **DP**：梯度 all-reduce 是 DDP 内部的异步操作（跑在自己的 comm stream 上），Python 无法计时；
  `analyze.py` 从 profiler 的 `ncclDevKernel_*` kernel 时长读 DP 的通信时间（`comm_kernel_ms_per_step`）。
- **TP/PP**：`CommTimer` 对阻塞的 all-reduce/send/recv 打 CUDA event，`analyze.py` 出 `comm_ms_per_step`。

两者都报，`summary_all.csv` 里 `comm_kernel_ms_per_step`（NCCL kernel 时间）是三模式可比的统一口径。

### nsys 交叉验证

torch.profiler 的时间轴 + `ncclDevKernel_*` 时长是主口径；nsys 用来交叉验证 kernel 级时延与
CPU 发射间隙/overlap：

```bash
bash launchers/nsys_wrap.sh dp 4 nsys_dp      # 生成 nsys_dp.nsys-rep
nsys stats nsys_dp.nsys-rep -r nvtx_sum,cuda_gpu_kern_sum   # 汇总 NVTX 相位 / GPU kernel
```

对同一个 all-reduce，nsys 的 `NCCL` kernel 时长应与 torch.profiler 的 `ncclDevKernel_*` 一致。

### 本地（Windows 单卡，无 NCCL）

```bash
# 单进程 GPU 冒烟（world=1，不初始化 dist，验证模型 + profiler）
python run.py --mode dp --steps 3 --hidden 256 --layers 2 --heads 8 --ffn 1024 --seq 64 --vocab 512

# 多进程逻辑校验（CPU + GLOO，绕过 torchrun 的 libuv 问题）
python launchers/local_launch.py --nproc 2 --port 29500 -- run.py --mode tp --backend gloo --device cpu --steps 3
python launchers/local_launch.py --nproc 2 --port 29501 -- run.py --mode pp --backend gloo --device cpu --steps 3
python launchers/local_launch.py --nproc 2 --port 29502 -- run.py --mode dp --backend gloo --device cpu --steps 3
```

> 本地环境：`F:\PycharmProjects\llm\.venv`（torch 2.12.1+cu126）。TP/PP 的 NCCL 通信只能
> 在远程多卡跑；本地用 GLOO+CPU 校验三者的**数学正确性**（TP 各 rank loss 一致、PP 末段
> loss == 整模型 loss）。

## 产物说明

每次运行在 `outputs/<tag>_<mode>_p<N>/` 下生成：

| 文件 | 内容 |
|---|---|
| `comm_rankR.jsonl` | 每 collective 一条：`{op, bytes, ms, gbps, rank, step}` |
| `steps_rankR.jsonl` | 每 step 端到端墙钟时延 |
| `trace/rankR.json` | torch.profiler 时间轴（TensorBoard） |
| `key_averages_rankR.{txt,csv}` | profiler 聚合表 |
| `meta.json` | 运行配置 |

`analyze.py` 输出每个 mode 的 `summary.csv`（逐 collective 的 count/avg_ms/p95/带宽），
以及跨 mode 的 `outputs/summary_all.csv` 和 `comm_comparison.png`（3 联图：comm/step、
有效带宽、通信占比）。

## 正确性自检

跑完后快速确认三件事，避免「profile 了一堆错误数据」：

1. **TP**：各 rank 的 loss 应**完全相同**（输出经 all-reduce 后复制）。
2. **PP**：只有末段 rank 有 loss，且与同配置 DP 的 loss **数值一致**。
3. **DP**：各 rank loss 正常（同种子下应一致）。
