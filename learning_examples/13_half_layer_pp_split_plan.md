# 半层切分（Half-Layer PP Split）实现方案

## Context

在 Attention/FFN 边界切分一个 Transformer Layer，将 Attention 部分和 FFN 部分分配给两个相邻的 PP Stage。

**切分点**：`transformer_layer.py:514` — `_forward_attention()` 返回 `(pre_mlp_layernorm_output, residual, context)` 之后，`_forward_mlp()` 被调用之前。

**跨 Stage 需要传递 2 个 tensor**：`pre_mlp_layernorm_output` [s,b,h] + `residual` [s,b,h]

**配置方式**：扩展 `--decoder-num-layers-per-pipeline-stage`，`pipeline_split_layers` 指定哪些层被切分。pre_mlp_layernorm 放在 Attention 侧。

---

## 改动文件清单

| 文件 | 改动 | 阶段 |
|------|------|------|
| `transformer/transformer_layer.py` | 提取 `_run_attention()` / `_run_mlp()` 模块级函数 | Phase 1 |
| `transformer/transformer_sublayer.py` | **新建** AttentionSubLayer + FFNSubLayer | Phase 1 |
| `transformer/transformer_block.py` | forward loop 3-way dispatch + input unpacking | Phase 1 |
| `models/gpt/gpt_layer_specs.py` | 切分感知的 per-stage spec 构建 | Phase 1 |
| `transformer/transformer_config.py` | `pipeline_split_layers` 字段 + 校验 | Phase 1 |
| `training/arguments.py` | CLI `--pipeline-split-layers` + 校验 | Phase 1 |
| `pipeline_parallel/schedules.py` | forward_step 归一化 + backward_step 多 tensor + get_tensor_shapes | Phase 1 |
| `models/gpt/gpt_model.py` | set_input_tensor 接受 2 tensors | Phase 1 |

---

## Phase 1: 最小可运行原型（PP=2, 1 个切分层, Dropout=0）

### Step 1: 提取 `_run_attention()` / `_run_mlp()` 

**文件**: `transformer/transformer_layer.py`

将 `TransformerLayer._forward_attention()` (L396-514) 和 `_forward_mlp()` (L516-560) 的 body 提取为模块级函数：

```python
def _run_attention(module, hidden_states, attention_mask=None, context=None, ...):
    # 原 _forward_attention 的 body，self.xxx → module.xxx
    # 返回 (pre_mlp_layernorm_output, residual, context)

def _run_mlp(module, pre_mlp_layernorm_output, residual):
    # 原 _forward_mlp 的 body，self.xxx → module.xxx
    # 返回 hidden_states
```

然后 `TransformerLayer._forward_attention` → `return _run_attention(self, ...)`，行为完全不变。

**风险**：必须 byte-identical，Phase 1 的 F.3 run C 验证无行为回归。

### Step 2: 新建 `AttentionSubLayer` / `FFNSubLayer`

**新文件**: `transformer/transformer_sublayer.py`

```python
class AttentionSubLayer(MegatronModule):
    """包含: input_layernorm → self_attention → self_attn_bda → pre_mlp_layernorm"""
    def __init__(self, config, submodules: TransformerLayerSubmodules, layer_number, ...):
        # 只构建 attention 侧的 submodule
    def forward(self, hidden_states, attention_mask, ...):
        return _run_attention(self, hidden_states, ...)
        # → (pre_mlp_layernorm_output, residual, context)

class FFNSubLayer(MegatronModule):
    """包含: mlp → mlp_bda"""
    def __init__(self, config, submodules, layer_number, ...):
        # 只构建 FFN 侧的 submodule
    def forward(self, pre_mlp_layernorm_output, residual):
        return _run_mlp(self, pre_mlp_layernorm_output, residual)
        # → hidden_states
```

关键细节：
- 两层都设置 `global_layer_number` 为被切分的原始层号
- `recompute_pre_mlp_layernorm` 在 AttentionSubLayer 中强制为 `False`（因为 checkpoint hook 的触发点 `discard_output_and_register_recompute` 在 FFN 侧，跨 stage 无法工作）
- 输出用 `make_viewless_tensor(..., requires_grad=True, keep_graph=True)` 包装

### Step 3: 修改 `TransformerBlock.forward()` 

**文件**: `transformer/transformer_block.py`

3a. **输入替换** (L478-480)：检测 `input_tensor` 是否为 2-tensor list（FFN 起始的 stage）

```python
pending_mlp_input = None
if not self.pre_process:
    if isinstance(self.input_tensor, list) and len(self.input_tensor) == 2:
        pending_mlp_input = (self.input_tensor[0], self.input_tensor[1])
        hidden_states = self.input_tensor[1]
    else:
        hidden_states = self.input_tensor if isinstance(...) else self.input_tensor[0]
```

3b. **Layer loop** (L530-549)：3-way dispatch

```python
for layer in self.layers:
    if isinstance(layer, AttentionSubLayer):
        pre_mlp, residual, context = layer(hidden_states=hidden_states, ...)
        pending_mlp_input = (pre_mlp, residual)
    elif isinstance(layer, FFNSubLayer):
        hidden_states = layer(*pending_mlp_input)
        pending_mlp_input = None
    else:
        hidden_states, context = layer(hidden_states=hidden_states, ...)
```

3c. **后处理**：如果最后一个 layer 是 AttentionSubLayer → 返回 2-tuple

### Step 4: Per-stage spec 构建

**文件**: `models/gpt/gpt_layer_specs.py` (`get_gpt_decoder_block_spec`)

在 `layer_specs[offset:offset+n]` 切片之前，对包含切分层边界的 stage 插入半层 spec：

- 上边界：如果上一 stage 的最后一个 layer 被切分 → 本 stage 开头插入 `FFNSubLayer` spec
- 下边界：如果本 stage 的最后一个 layer 被切分 → 替换为 `AttentionSubLayer` spec
- 半层的 `global_layer_number` 都设置为原始 layer index

### Step 5: 配置 + CLI

**文件**: `transformer/transformer_config.py` + `training/arguments.py`

```python
# transformer_config.py
pipeline_split_layers: Optional[List[int]] = None  
# 被切分的 layer 编号（1-based global index）

# arguments.py
--pipeline-split-layers 6  # 将 layer 6 切成两半
```

校验：
- `pipeline_split_layers` 必须依赖 `decoder_num_layers_per_pipeline_stage`
- 每个切分索引必须是 stage 边界（即 `sum(decoder_num_layers_per_pipeline_stage[:r+1])`）
- 禁止 0 或 num_layers
- Phase 1 禁止 VPP, full recompute, CUDA graph, cpu_offloading, deallocate_pipeline_outputs

### Step 6: PP Schedule 改动

**文件**: `pipeline_parallel/schedules.py`

6a. `forward_step` (L277)：normalize 模型返回的 tuple 为 flat list（避免 `[(t1,t2)]` 嵌套）

6b. `backward_step` (L395)：改为 `torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)`（接受 multi-tensor list，单 tensor 退化为原行为）

6c. `get_tensor_shapes` (L1638)：在 split boundary 的 rank 上 append 第二个 shape（和第一个相同 `[s,b,h]`）。复用现有的 list-of-shapes 机制（encoder-decoder 已有先例）。

### Step 7: GPTModel.set_input_tensor

**文件**: `models/gpt/gpt_model.py:219-233`

```python
assert len(input_tensor) in (1, 2), 'input_tensor must have length 1 or 2'
self.decoder.set_input_tensor(input_tensor)  # 透传 list
```

---

## 验证计划

### F.1 Smoke test (CPU, mock parallel_state)

`learning_examples/half_layer_smoke.py`：PP=2, 4 layers, split layer 2。Mock parallel_state，验证 stage 0 的 layers = `[TransformerLayer(1), AttentionSubLayer(2)]`，stage 1 = `[FFNSubLayer(2), TransformerLayer(3), TransformerLayer(4)]`。对比单卡完整模型 → forward/backward 输出一致。

### F.2 Distributed GPU check

`learning_examples/half_layer_validate.py`：`torchrun --nproc_per_node=2`，12 layers, split layer 6。验证：stage 0 输出 2-tuple、stage 1 接收 2 tensors、backward 两个 input grad 都非 None。

### F.3 End-to-end 等价性

- Run A: PP=1, dropout=0, seed fixed → 基线 loss curve
- Run B: PP=2, split layer 6, 同 seed → loss curve 必须完全一致
- Run C: PP=2, 无 split → 行为无回归

### F.4 负向测试

split 不在 stage boundary、split layer 0、split + VPP 等 → 每个都报清晰错误。

---

## Phase 2/3（后续）

| 阶段 | 内容 |
|------|------|
| Phase 2 | 多切分层、PP>2、通用 per-stage spec 构建、selective recompute |
| Phase 3 | Full recompute、deallocate_pipeline_outputs、TE 兼容、VPP、CUDA graph、checkpoint 转换 |
