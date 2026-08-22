# 多卡分布式训练 Profiling — 目标与计划

## 目标

在**远程多卡**环境上，对一个**经典 transformer**（mini GPT，单机多卡即可）分别跑
**DP / TP / PP** 三种并行策略，采集并分析多卡通信信息，回答：

1. **同步通信时机**：每种策略在训练一个 step 内，何时、以什么顺序、多少次发生
   all-reduce / P2P send-recv（同步点的数量与位置）。
2. **量化的通信带宽与时延**：每种 collective 的字节量、单次时延（均值/P95）、有效
   带宽（GB/s）。
3. **通信占 step 比例**：每种策略通信时间占端到端 step 时间的百分比。

## 方案

一个**自包含**的 `profiling/` 工程（不依赖 Megatron 内部），核心是一个 TP-native 的
mini GPT transformer（`tp_size==1` 时即普通模型），加上三种并行实现：

| 模式 | 实现 | 通信特征（每 step） |
|---|---|---|
| **DP** | `DDP` + 自定义 comm hook | 反向结束时按梯度桶 **all-reduce**（消息大、次数少） |
| **TP** | Megatron 风格 Column/RowParallelLinear | 每层 **2 前向 + 2 反向 all-reduce**（消息为隐藏层维度的切片） |
| **PP** | Gpipe 调度 + 阻塞 `send`/`recv` | 每 microbatch 前向/反向各一次 **P2P**（激活/梯度） |

TP 的 `ColumnParallelLinear` 前向不通信、反向对输入梯度 all-reduce；
`RowParallelLinear` 前向对输出 all-reduce、反向不通信。这两个 `autograd.Function`
把反向通信也纳入 profile（PyTorch 原生 autograd 无法表达分布式求和）。

### 两种采集机制（互补）

1. **`torch.profiler`**（默认开）：导出每 rank 的 TensorBoard timeline
   （`trace/rankN.json`，含 NCCL kernel、时间轴、前向/反向/优化器相位）+
   `key_averages` 表。这是「通信时机/重叠」的权威来源，用 TensorBoard 看时间轴。
2. **`CommTimer`**（默认开）：给每一个 collective 打 CUDA-event 计时，输出
   `comm_rankN.jsonl`（op、bytes、ms、gbps）。这是「带宽/时延」量化的权威来源，
   由 `analyze.py` 汇总。

> 计时方式（v2）：**阻塞** collective（TP all-reduce、PP send/recv）用 CUDA event 包裹，
> 事件逐 op 记录、flush 时一次性 `synchronize` 结算——不再每 op 插同步栅栏；**异步** all-reduce
> （DDP 梯度桶）跑在 DDP 自己的通信流上，无法用计算流上的 event 包裹，改用「启动→future 完成」
> 的墙钟计时，从而**保留 DDP 原生的反向/通信重叠**。

### 关键实现约定

- **all-reduce 有效字节**：`2*(n-1)/n * numel * elem_size`（ring 算法近似，n 为组大小）。
- **P2P 字节**：`numel * elem_size`。
- **带宽**：`bytes / 1e9 / (ms / 1e3)`。
- **comm/step**：对每个 (rank, step) 求和该 rank 该步所有 collective 时延，再跨 rank/step
  取均值（warmup 步已剔除）。
- 数据为**同种子确定性合成数据**，各 rank 一致，保证 TP 各 rank loss 一致、PP 末段 loss
  等于整模型 loss，可作正确性自检。

## 本地已验证（Windows，无 NCCL）

本地只有单卡 + torch 无 libuv，采用两层验证：

| 验证 | 命令 | 结论 |
|---|---|---|
| 单进程 GPU 冒烟（dp/tp/pp, `world=1`） | `python run.py --mode {dp,tp,pp} ...` | 三种模式前向/反向/优化器 + profiler 导出正常 |
| **TP 正确性**（2 进程 CPU GLOO） | `local_launch.py --nproc 2 --mode tp --backend gloo --device cpu` | 两 rank loss 完全一致，前/反向 all-reduce 数学正确 |
| **PP 正确性**（2 进程 CPU GLOO） | `local_launch.py --nproc 2 --mode pp ...` | 末段 loss 与整模型 DP loss 数值一致，Gpipe 梯度传播精确 |
| **DP 正确性**（2 进程 CPU GLOO） | `local_launch.py --nproc 2 --mode dp ...` | comm hook 记录梯度桶 all-reduce，loss 正常 |

`local_launch.py` 是 Windows 专用的轻量启动器（绕过 torchrun elastic rendezvous 的
libuv 问题）；远程 Linux 用 `torchrun` 即可。

## 远程执行步骤

1. 上传 `profiling/` 到服务器，确认 `torch`（CUDA 版，带 NCCL）可用。
2. 三种模式各跑一次（同模型配置以便对比）：
   ```bash
   bash launchers/run_remote.sh dp 4 cmp
   bash launchers/run_remote.sh tp 4 cmp
   bash launchers/run_remote.sh pp 4 cmp
   ```
3. 用 nsys 抓一次 timeline（可选，作为 torch.profiler 的交叉验证）：
   ```bash
   bash launchers/nsys_wrap.sh pp 4 nsys_pp
   ```
4. 下载 `outputs/` 到本地，跑分析：
   ```bash
   python analyze.py            # 汇总表 + CSV + 对比图
   tensorboard --logdir outputs/cmp_pp_p4/trace   # 看时间轴
   ```

## 分析口径（读表须知）

- **通信占比高 ≠ 一定差**：TP 每步 8 个同步点但消息小、且与计算重叠空间有限；DP 每步
  2 个同步点但消息大；PP 的 P2P 在 Gpipe 下是「裸暴露」的串行气泡（1F1B 才会隐藏）。
- **带宽随消息大小变化**：小消息（TP 切片、PP 激活）带宽低是正常的——受时延主导，
  远达不到大消息（DP 全梯度）的峰值带宽。对比时应关注「同字节量」下的带宽，而非跨策略直接比 GB/s。
- **单次时延 vs 总通信时间**：`avg_ms` 是单次 collective 时延，`comm/step` 是次数×时延的
  总和。两者合起来才说明「同步时机」的代价。
- warmup 步已从汇总中剔除（`meta.json` 里的 `warmup`）。

## 远程 p4 实测解读（4×V100-PCIe-16GB，hidden=1024/layers=8/seq=512，FP32）

硬件从 trace 的 `deviceProperties` 确认是 **Tesla V100-PCIe**（compute 7.0）：卡间走
**PCIe 而非 NVLink**，且 FP32 无 tensor core（`volta_sgemm_*` 跑 CUDA core）。

| 模式 | step | NCCL kernel 时间/步 | 占比 | kernel 数/步 | 单次时延 |
|---|---:|---:|---:|---:|---:|
| DP | ~335ms | ~271ms | 65.6% | 14 | 19.3ms |
| TP | ~370ms | ~215ms | 56.8% | 32 | 6.4ms |
| PP | ~204ms | ~110ms | 58.6% | 12 | send 0.4~3ms / recv 11~27ms |

结论：

1. **`autograd::engine::evaluate_function` 占比高是正常的**（反向分发器父节点，Self CPU 仅
   ~0.65%）；修复 sync 后它从 ~44% 降到 ~3%，说明之前的高占比是同步测量塞进去的。
2. **NCCL all-reduce 单次 19.3ms = ~2.5 GB/s，是 V100-PCIe 的硬件现实**，不是测量误差。
   DP 的梯度 all-reduce（每桶 47.6MB）在 PCIe 上就是这么慢。它**与反向重叠**，所以 DP 的
   wall-clock step 仍是 compute-bound（~335ms），通信「藏」在 compute 下面。
3. **PP 的 recv 方差极大（`recv_grad` p95~82ms vs `send_grad` 0.4ms）是 Gpipe 气泡**：recv
   阻塞等下游算完，正是流水线停顿的量化体现。压气泡要上 1F1B/VPP。
4. **FP32 无 tensor core，compute 偏慢，把通信盖住了**；换 `--dtype fp16`（V100 tensor core）
   或 `bf16`（Ampere+）后 compute 大幅缩短，通信会变成瓶颈——这才是研究通信要看的真实场景。
5. DP 通信时间改从 profiler 的 `ncclDevKernel_*` 读（`comm_kernel_ms_per_step`），DP 的
   `CommTimer` 墙钟计时不可靠（future 在 kernel 完成前就 resolve），已移除。

## 局限与后续

## 局限与后续

- 当前为**单维并行**（每模式用满 world_size）；3D（DP×TP×PP）组合需额外进程组，留作扩展。
- PP 用最简单的 **Gpipe**（无 1F1B 重叠）；1F1B / VPP 的通信时机与气泡结构不同，可作为
  后续对照。
- `torch.profiler` 的 NCCL kernel 时延可与 `CommTimer` 相互印证；`nsys` 提供 NVTX 相位的
  CPU 发射间隙/overlap 视角。
- 若要「同字节量带宽曲线」，可加一个纯 collective 微基准（`all_reduce`/`send-recv` 扫
  不同消息大小），与训练 profile 的实测带宽对照。
