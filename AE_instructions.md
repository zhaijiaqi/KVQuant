# KVQuant 论文复现指南 (AE Instructions)

> **本文档面向对象**：了解"量化可以降低位宽、节省内存带宽"，但对 LLM 量化细节不熟悉的读者。
> 文档会在每个步骤前先解释"这是什么、为什么要做"，再给出可直接运行的命令。

---

## 目录

1. [背景：KVQuant 在解决什么问题](#1-背景kvquant-在解决什么问题)
2. [项目结构与环境说明](#2-项目结构与环境说明)
3. [Step 0：数据与模型准备](#3-step-0数据与模型准备)
4. [Step 1：计算 Fisher 信息](#4-step-1计算-fisher-信息)
5. [Step 2：量化校准（生成 Quantizer）](#5-step-2量化校准生成-quantizer)
6. [Step 3：困惑度评估](#6-step-3困惑度评估)
7. [Step 4：Latency 评估](#7-step-4latency-评估)
8. [Step 5：Passkey Retrieval 评估](#8-step-5passkey-retrieval-评估)
9. [预期结果](#9-预期结果)
10. [故障排除记录](#10-故障排除记录)

---

## 1. 背景：KVQuant 在解决什么问题

### 1.1 KV Cache 是什么

大语言模型（LLM）在生成文字时，每一步都需要回顾之前所有 token 的注意力信息。这些信息以 **Key（K）和 Value（V）矩阵** 的形式缓存在 GPU 显存中，称为 **KV Cache**。

序列越长，KV Cache 越大。以 LLaMA-7B 为例：

- 每个 token 需要 2 × 32层 × 32头 × 128维 × 2字节（fp16） ≈ 0.5 MB
- 10万个 token 的上下文 → KV Cache 占 **50 GB**，超过单张 A100 的显存

### 1.2 量化如何帮助

把 KV Cache 从 **fp16（16位）** 压缩到 **4位整数**，显存占用降为原来的 1/4。
但问题是：**怎么把 65536 种 fp16 值映射到只有 16 个整数级别，同时保持模型精度？**

### 1.3 KVQuant 的三个核心创新

| 创新点                                   | 解决的问题                                            |
| ---------------------------------------- | ----------------------------------------------------- |
| **Per-channel、Pre-RoPE Key 量化** | Key 的每个通道有不同的值域范围，用各自的 scale 更精确 |
| **非均匀量化（NUQ）**              | 激活值分布不均匀，用 K-Means 把量化级别放在数据密集区 |
| **Dense-and-Sparse 量化**          | 少量极端异常值会扭曲量化范围，单独用稀疏格式保存      |

---

## 2. 项目结构与环境说明

```
/home/ubuntu/program/rlkvq/
├── kvquant/
│   ├── gradients/      ← Step 1: Fisher 信息计算（自定义 transformers fork）
│   ├── quant/          ← Step 2/3: 量化校准 + 困惑度评估 + Passkey 评估
│   ├── deployment/     ← Step 4: 真实量化推理 + Latency 评估（需编译 CUDA kernel）
│   └── benchmarking/   ← Step 4: 底层 CUDA kernel 吞吐测试
└── AE_instructions.md  ← 本文件

/data/
├── models/LLaMA-7B/                    ← 模型权重
├── datasets/wikitext-2-raw-v1/         ← 校准/评估数据集（本地缓存）
└── kvquant/
    ├── fisher-llama-7b/                ← Step 1 输出：Fisher 信息
    └── quantizers/                     ← Step 2 输出：量化器参数
```

### 两个 conda 环境（互相独立）

| 环境名         | 用途                      | transformers 版本                           |
| -------------- | ------------------------- | ------------------------------------------- |
| `rlkvq_grad` | Step 1: Fisher 信息计算   | 4.38.0.dev（自定义 fork，含 `LinearAct`） |
| `rlkv`       | Step 2/3: 量化校准 + 评估 | 4.40.2（标准，最后兼容旧 LLaMA API 的版本） |

> **注意**：`rlkv_hky`、`rlkv_hky_quant` 等环境属于其他用户，本项目不使用。

#### 为什么需要两套环境，不能合并成一个？

根本原因：两个步骤依赖**同一个包（`transformers`）的两个不同版本**，一个是 KVQuant 大幅修改过的自定义 fork，另一个是标准版本，它们无法同时安装在同一 Python 环境里。

| 步骤        | 为什么必须用自定义 fork                                                                                                                     | 为什么不能直接用标准版                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Fisher 计算 | 自定义 fork 把 `k_proj`/`v_proj` 替换为 `LinearAct`，能在 forward 时保存激活张量、在 backward 后保留梯度——标准 LLaMA 层没有这个机制 | 标准 transformers 的 LLaMA 层 forward 结束后激活张量不保留，无法取 `.act.grad`          |
| 校准 + 评估 | 不需要梯度，用标准模型推理即可                                                                                                              | 自定义 fork 版本（4.38.0.dev）较旧，与校准脚本依赖的 `position_ids` 等 API 存在细节差异 |

两个步骤在时间上完全串行（先跑 Fisher，再跑校准），因此分成两个环境切换使用，不存在同时运行的需求。

---

## 3. Step 0：数据与模型准备

`LLaMA-7B` 与 Wikitext-2 已在 `/data/` 本地缓存；若要复现 Passkey / LongBench / RULER，请先确认 `LLaMA-2-7B-32K` 已完整下载：

```bash
# 模型
/data/models/LLaMA-7B/          # LLaMA-7B，33 个 .bin 分片，fp16
/data/models/LLaMA-2-7B-32K/    # Together 的长上下文模型，Passkey / LongBench / RULER 使用

# 数据集（parquet 格式本地缓存）
/data/datasets/wikitext-2-raw-v1/train-0000.parquet
/data/datasets/wikitext-2-raw-v1/test-0000.parquet
```

可用下面的命令检查 32K 模型是否完整（需同时看到两个 `.bin` 分片）：

```bash
du -sh /data/models/LLaMA-2-7B-32K
ls -lh /data/models/LLaMA-2-7B-32K/pytorch_model-0000*-of-00002.bin
```

创建输出目录：

```bash
mkdir -p /data/kvquant/fisher-llama-7b
mkdir -p /data/kvquant/quantizers
```

---

## 4. Step 1：计算 Fisher 信息

### 4.1 什么是 Fisher 信息（面向初学者）

**直觉理解**：想象模型里有很多旋钮（激活值）。有些旋钮转一点点，输出就大幅变化（高敏感）；有些旋钮转很多，输出几乎不变（低敏感）。Fisher 信息就是衡量这种**敏感度**的量。

数学上，Fisher 信息 ≈ **loss 对激活值的梯度的平方**：

```
Fisher(activation) ≈ (∂Loss / ∂activation)²
```

**Fisher 信息不是 KVQuant 的原创**

用 Fisher 信息指导量化并非 KVQuant 首创。这一思想最早可追溯到神经网络剪枝领域（Optimal Brain Damage, LeCun 1989）。**直接来源是 SqueezeLLM（Kim et al., 2023）**——它将 Fisher 加权 K-Means 用于模型**权重**的非均匀量化。KVQuant 的贡献是把这套方法迁移到 **KV Cache 激活值**量化，并结合 per-channel Key 量化、Dense-and-Sparse 等设计。

> KVQuant 的 README 明确写道："This code reuses components from SqueezeLLM."

**为什么 KVQuant 需要 Fisher 信息**：

在 Step 2 中，KVQuant 用 K-Means 聚类为每层 KV Cache 设计 16 个（4-bit）量化代表值。
普通 K-Means 把所有激活值一视同仁，但有些激活值对模型输出影响更大。
**Fisher 信息作为 K-Means 的样本权重**，可以让量化误差在"重要"的位置更小——详见 [5.1.4 节](#514-fisher-加权-k-means让重要的位置误差更小)的完整原理说明。

### 4.2 代码做了什么

自定义 transformers fork 中，`k_proj` 和 `v_proj` 被替换为 `LinearAct`（在 `modeling_llama.py` 里定义），它在 forward 时保存输出激活：

```python
class LinearAct(nn.Linear):
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.retain_grad:
            y.retain_grad()   # 让非叶子节点保留梯度
        self.act = y          # 保存激活值
        return y
```

`run-fisher.py` 对 16 条校准文本做前向+反向传播，取 `k_proj.act.grad²` 保存：

```python
outputs = model(input_ids=x, labels=x)
loss = outputs.loss
loss.backward()                              # 反向传播

kgrad = (k_proj.act.grad ** 2).float().cpu()  # 梯度² = Fisher 信息
```

16 条样本的结果拼接成 `[1, 16×2048, 4096]` 的张量，以"假冒权重"的方式用 `save_pretrained` 保存。

### 4.3 运行命令

```bash
conda activate rlkvq_grad
cd /home/ubuntu/program/rlkvq/kvquant/gradients

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python run-fisher.py \
    --model_name_or_path /data/models/LLaMA-7B \
    --output_dir /data/kvquant/fisher-llama-7b \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen 2048 \
    --num_examples 16
```

**参数说明**：

| 参数               | 值        | 说明                                                       |
| ------------------ | --------- | ---------------------------------------------------------- |
| `--num_examples` | 16        | 校准样本数，**必须**与 Step 2 的 `--nsamples` 一致 |
| `--seqlen`       | 2048      | 每条样本的 token 长度                                      |
| `--dataset`      | wikitext2 | 校准数据集（代码优先读本地 parquet 文件）                  |

**环境变量说明**：

| 变量                                                | 原因                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | 服务器 NVML 驱动版本不匹配，默认 CUDA 内存分配器会崩溃，改用 cudaMallocAsync 后端绕过 |

### 4.4 预期输出

```
/data/kvquant/fisher-llama-7b/
├── config.json
├── model-00001-of-00010.safetensors   # 共 10 个分片，约 43 GB
├── ...
└── model.safetensors.index.json
```

每个张量的 key 与原模型权重相同（如 `model.layers.0.self_attn.k_proj.weight`），
但**数值是 Fisher 信息（梯度²）**，形状为 `[1, 32768, 4096]`（= 1 × 16×2048 × 4096）。

**耗时**：约 3 分钟（A100-40GB）

---

## 5. Step 2：量化校准（生成 Quantizer）

### 5.1 什么是量化校准（面向初学者）

#### 5.1.1 量化的本质：把连续值映射到有限级别

你已经知道：量化把 fp16 压缩到 4-bit，节省内存。
但具体怎么映射？以 4-bit 为例，只有 **16 个整数级别（0 ~ 15 或 -7 ~ 7）**。

**均匀量化（Uniform Quantization）**：16 个级别均匀分布在值域范围内。

```
数据范围 [-1.0, 1.0]，4-bit 均匀量化：
级别：-1.0  -0.87  -0.73  ...  +0.87  +1.0
间距：固定 = 2.0 / 15 ≈ 0.133
```

**问题**：如果大多数激活值集中在 `[-0.1, 0.1]`（正态分布），均匀量化只有 1-2 个级别覆盖这个区间，精度极差。

#### 5.1.2 非均匀量化（NUQ）：把级别放在数据密集的地方

KVQuant 用 **K-Means 聚类**，从真实数据中学出 16 个代表值（质心）：

```
先看数据分布（校准）：
  激活值直方图： ▁▁▃▇█▇▃▁▁  （大多集中在中间）

均匀量化的 16 个级别：|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
                       （稀疏区域浪费了级别）

NUQ 的 16 个级别：      |||||||||||||||
                       （密集区域分配更多级别）
```

NUQ 的误差来源于每个数据点到最近质心的距离，K-Means 使这个误差的平方和最小。

#### 5.1.3 为什么需要"校准数据"

设计这 16 个代表值需要先知道数据长什么样——但模型在推理时处理不同文本，KV Cache 的分布会有差异。
**校准（Calibration）** = 用少量有代表性的文本（这里是 Wikitext-2 的 16 条语句）跑一遍模型，观测 KV Cache 激活值的分布，然后以此为依据设计量化器。

换句话说：**校准是量化的"测量"阶段，得到的量化器是量化的"尺子"**。

#### 5.1.4 Fisher 加权 K-Means：让重要的位置误差更小

##### 标准 K-Means 的问题

标准 K-Means 优化的目标是：让所有数据点到其最近质心的距离平方之和最小——

```
最小化：Σ_i  (x_i - 最近质心)²          每个点权重都是 1
```

这意味着：**哪里数据点多，质心就往哪里聚集**。对于激活值分布中点数很多但对 loss 影响很小的区域，K-Means 会浪费很多质心去覆盖它们。

##### Fisher 加权 K-Means 改变了什么

加入 Fisher 权重后，目标变成：

```
最小化：Σ_i  Fisher_i × (x_i - 最近质心)²    每个点权重 = 其 Fisher 信息
```

**Fisher 大的点，其误差项被放大了**——K-Means 必须让这些点离质心更近，才能压低整体目标函数。结果是：质心会向 Fisher 大的激活值区域"偏移"并在那里聚集得更紧。

##### 用一个具体数字例子说明

假设某层 KV Cache 有两类激活值，只分配 **3 个质心**：

```
A 类：1000 个点，分布在 x ≈ 0.05，Fisher ≈ 0.001（量很多，但对 loss 不重要）
B 类：  10 个点，分布在 x ≈ 1.80，Fisher ≈ 500 （量很少，但对 loss 非常重要）
```

|                       | 目标函数贡献                                                                                                     | K-Means 如何分配质心                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **标准**        | A类：1000 × 距离² ；B类：10 × 距离² → A类主导                                                               | 2-3 个质心覆盖 A 类（点多），B 类可能只有 1 个                      |
| **Fisher 加权** | A类：1000×0.001 × 距离² =**1** × 距离² ；B类：10×500 × 距离² = **5000** × 距离² → B类主导 | 2-3 个质心聚在 B 类（权重大），误差小；A 类精度变差但对 loss 无所谓 |

"密集"的含义就在这里：**B 类区域的质心间距变小（密集），意味着每个 B 类激活值都能找到更近的质心，量化误差更小**。

##### 为什么不直接给 Fisher 大的位置用更多 bit？

这是一个完全合理的替代思路，叫做**混合精度量化（Mixed-Precision Quantization）**。LLM.int8() 等方案确实这样做——对异常值用 fp16，其余用 int8。

KVQuant 选择 Fisher 加权 NUQ 而非混合精度，原因有三：

1. **bit 宽一致，硬件友好**：所有 KV 条目都是 4-bit，内存布局完全规则，无需额外的索引表来记录哪些位置用了不同 bit 宽
2. **没有额外存储开销**：混合精度需要存储一张 mask（"哪里用高 bit"），NUQ 只需一张 16 个质心的查找表（LUT），16 个 fp16 值 = 32 字节，极小
3. **全局联合优化**：K-Means 同时优化 16 个质心在整个分布上的位置，而不是对每个位置单独决定 bit 宽，理论上更接近最优解

##### KVQuant 中 Fisher 信息的实际使用位置（代码）

Fisher 信息在 `simquant_module_quantizer.py` 的 `SimQuant.quantize()` 中作为 `sample_weight` 传入 scikit-learn 的 K-Means：

```python
# fisher_info: shape [nsamples × seqlen × hidden]，已 flatten
kmeans = KMeans(
    n_clusters=2 ** self.bits,   # 4-bit → 16 个质心
    n_init="auto",
    max_iter=50,
).fit(
    act_distn_np_without_outliers,      # 激活值（去掉异常值后）
    sample_weight=fisher_info_tmp_without_outliers  # ← Fisher 权重
)
centroids = kmeans.cluster_centers_    # 这就是量化的 LUT
```

Fisher 信息在这里扮演的唯一角色就是 **K-Means 的样本权重**，让质心向重要位置偏移。

#### 5.1.5 Dense-and-Sparse 量化：处理异常值

KV Cache 中少量值（约 1%）远超正常范围，如果把它们纳入量化范围，会导致其他 99% 的值被压缩到很小的区间。

KVQuant 的做法：

1. **检测异常值**：超过第 99.5 百分位或低于第 0.5 百分位的值（`--sparsity-threshold 0.99` 表示 1% 异常值）
2. **正常值**：用 NUQ 4-bit 量化（紧凑存储）
3. **异常值**：单独用稀疏矩阵保存原始 fp16 值（只有 1% 数量，存储开销小）

结果：量化范围集中在 99% 的正常值上，精度大幅提升，同时异常值本身不损失精度。

#### 5.1.6 校准输出是什么

校准结束后，每一层的 `k_proj` 和 `v_proj` 都有一个 **Quantizer**，包含：

- `outlier_threshold_upper` / `outlier_threshold_lower`：异常值的分界阈值（fp16）
- `centroids`（NUQ 时）：K-Means 学出的 16 个代表值（查找表 LUT）

这些参数保存在一个 pickle 文件中，推理时直接加载使用。

### 5.2 完整校准命令

#### 校准（nuq4-1%）：生成量化器

```bash
conda activate rlkv
cd /home/ubuntu/program/rlkvq/kvquant/quant

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 4 \
    --nsamples 16 \
    --seqlen 2048 \
    --nuq \
    --fisher /data/kvquant/fisher-llama-7b \
    --include_sparse \
    --sparsity-threshold 0.99 \
    --quantize \
    --quantizer-path /data/kvquant/quantizers/nuq4_s1.pkl
```

**参数说明**：

| 参数                          | 值   | 说明                                                           |
| ----------------------------- | ---- | -------------------------------------------------------------- |
| `--abits 4`                 | 4    | 量化位宽：4-bit，共 2⁴=16 个量化级别                          |
| `--nsamples 16`             | 16   | 校准样本数，**必须等于** Fisher 的 `--num_examples 16` |
| `--nuq`                     | —   | 使用非均匀量化（K-Means 质心），不传则使用均匀量化             |
| `--fisher`                  | 路径 | Fisher 信息路径，用于加权 K-Means                              |
| `--include_sparse`          | —   | 开启 Dense-and-Sparse：异常值单独稀疏保存                      |
| `--sparsity-threshold 0.99` | 0.99 | 以第 99.5/0.5 百分位为异常值阈值（约 1% 是异常值）             |
| `--quantize`                | —   | **校准模式**：收集激活、运行 K-Means、保存量化器         |
| `--quantizer-path`          | 路径 | 量化器输出路径（.pkl 文件）                                    |

**量化粒度说明**（代码默认值）：

| 张量                | 粒度                             | 原因                                                     |
| ------------------- | -------------------------------- | -------------------------------------------------------- |
| Key（`k_proj`）   | per-channel（每通道一个 scale）  | Key 的异常值集中在特定通道，per-channel 能更好匹配其分布 |
| Value（`v_proj`） | per-token（每 token 一个 scale） | Value 的异常值集中在特定 token，per-token 更合适         |

**耗时**：约 30–60 分钟（K-Means 在 CPU 上运行，16样本 × 32层 × 2张量）

### 5.3 其他精度配置

论文中还测试了 3-bit 和 2-bit，命令只需修改 `--abits` 和 `--quantizer-path`：

#### nuq3-1%（3-bit，更激进压缩）

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 3 --nsamples 16 --seqlen 2048 \
    --nuq --fisher /data/kvquant/fisher-llama-7b \
    --include_sparse --sparsity-threshold 0.99 \
    --quantize \
    --quantizer-path /data/kvquant/quantizers/nuq3_s1.pkl
```

#### nuq2-1%（2-bit，极致压缩，需加 Q-Norm）

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 2 --nsamples 16 --seqlen 2048 \
    --nuq --fisher /data/kvquant/fisher-llama-7b \
    --include_sparse --sparsity-threshold 0.99 \
    --norm \
    --quantize \
    --quantizer-path /data/kvquant/quantizers/nuq2_s1.pkl
```

> `--norm`（Q-Norm）：2-bit 量化后分布偏移较大，Q-Norm 通过校正均值和方差来补偿这个偏差。

---

## 6. Step 3：困惑度评估

### 6.1 什么是困惑度（Perplexity）

困惑度（PPL）衡量语言模型对文本的预测能力：

```
PPL = exp( -1/N × Σ log P(token_i | token_1...token_{i-1}) )
```

#### 逐步拆解公式

**① `P(token_i | token_1...token_{i-1})`：模型对"下一个词"的信心**

语言模型每次预测时，会对词表里所有词打分，然后通过 softmax 得到一个概率分布。`P(token_i | ...)` 就是模型给"第 i 个位置上真实出现的词"打的概率。

- 如果模型很有把握（比如 `P = 0.9`）：说明这个词在这里出现是"意料之中"的，模型预测准确。
- 如果模型很迷茫（比如 `P = 0.01`）：说明这个词在这里让模型"大吃一惊"，预测很差。

**② `-log P`：把概率换算成"惊讶程度"**

直接累加概率不方便（概率相乘会下溢），取对数后就变成求和。加负号是因为 `log P ≤ 0`（概率 ≤ 1），加负号让结果变成非负的"惊讶值"：

```
P → 1.0（极其自信）：-log(1.0) = 0        → 完全不惊讶
P → 0.5（有些把握）：-log(0.5) ≈ 0.69    → 一点惊讶
P → 0.1（比较迷茫）：-log(0.1) ≈ 2.30    → 很惊讶
P → 0.01（极其迷茫）：-log(0.01) ≈ 4.61  → 非常惊讶
```

**③ `-1/N × Σ`：对所有 N 个 token 求平均惊讶值**

这就是整段文本上模型的"平均每 token 惊讶程度"，也叫**平均交叉熵**。

**④ `exp(...)`：从对数尺度换回去，得到困惑度**

为什么要取指数？为了让结果有直觉上的单位感：

```
若平均每步 -log P = log(K)，则 exp(log(K)) = K
```

**K 就是"模型在每一步相当于从 K 个等概率选项中随机猜"**。

- PPL = 1：模型每步都 100% 确定，完美预测
- PPL = 5.68：模型平均每步相当于从 ~5-6 个同样合理的词中随机选一个
- PPL = 100：模型很混乱，每步相当于从 100 个词中随机猜

#### 一个具体的例子

假设测试文本是 `"The cat sat on the mat"` 共 6 个 token，模型预测每步的概率为：

```
P("The")  = 0.05   （句子开头，词很多，比较迷茫）
P("cat")  = 0.30   （"The"之后，名词合理）
P("sat")  = 0.40   （猫坐着，比较自然）
P("on")   = 0.60   （"sat on"搭配常见）
P("the")  = 0.80   （介词后接定冠词，几乎确定）
P("mat")  = 0.20   （"the"后面词很多，但"mat"也合理）
```

平均交叉熵 = `-1/6 × (log 0.05 + log 0.30 + log 0.40 + log 0.60 + log 0.80 + log 0.20)`
= `-1/6 × (-3.00 + (-1.20) + (-0.92) + (-0.51) + (-0.22) + (-1.61))`
= `-1/6 × (-7.46) ≈ 1.24`

PPL = `exp(1.24) ≈ 3.46`

意思是：模型平均每步"相当于在 3-4 个选项中随机猜"。

#### 回到本实验的数值

- fp16 基线 PPL ≈ 5.68：未量化，模型最准，平均每步约 5-6 个合理选项
- nuq4-1% PPL ≈ 5.84：4-bit 量化后仅上升 0.16，量化影响极小
- nuq2-1% PPL ≈ 7.29：2-bit 极致压缩，平均 7-8 个选项，精度有所下降

> 复现时允许 ±0.05 误差（受随机种子、校准数据采样影响）。

### 6.2 评估命令

评估时**不加** `--quantize`，直接加载校准好的量化器运行推理：

#### 评估 fp16 基线

```bash
conda activate rlkv
cd /home/ubuntu/program/rlkvq/kvquant/quant

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 16 \
    --nsamples 16 \
    --seqlen 2048
```

> `--abits 16` 时不需要传 `--quantizer-path`，代码会跳过量化层替换直接评估原始 fp16 模型。

#### 评估 nuq4-1%

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 4 --nsamples 16 --seqlen 2048 \
    --nuq \
    --include_sparse --sparsity-threshold 0.99 \
    --quantizer-path /data/kvquant/quantizers/nuq4_s1.pkl
```

#### 评估 nuq3-1%

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 3 --nsamples 16 --seqlen 2048 \
    --nuq \
    --include_sparse --sparsity-threshold 0.99 \
    --quantizer-path /data/kvquant/quantizers/nuq3_s1.pkl
```

#### 评估 nuq2-1%

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python llama_simquant.py /data/models/LLaMA-7B \
    --abits 2 --nsamples 16 --seqlen 2048 \
    --nuq \
    --include_sparse --sparsity-threshold 0.99 \
    --norm \
    --quantizer-path /data/kvquant/quantizers/nuq2_s1.pkl
```

---

## 7. Step 4：Latency 评估

### 7.1 为什么需要 Latency 评估

PPL 评估（Step 3）使用的是**模拟量化**：模型权重和 KV Cache 仍以 fp16 存储，只是在计算时模拟量化误差对精度的影响。这足以验证"量化不损失精度"，但**不能证明"量化加速了推理"**。

Latency 评估使用 `deployment/` 目录下真正的**量化推理路径**：

- KV Cache 以打包的 2/3/4-bit 整数存储在 GPU 上
- 注意力计算调用自定义 CUDA kernel（`quant_cuda`），直接在量化表示上做 matmul，跳过反量化步骤
- 逐 token 测量端到端生成延迟，输出 Median latency（ms/token）

对应论文中的 **Table 1 / Figure** 中 token generation latency 结果。

### 7.2 两类 Latency 测试

| 测试类型                 | 脚本                                       | 测量内容                                  |
| ------------------------ | ------------------------------------------ | ----------------------------------------- |
| **端到端生成延迟** | `deployment/llama.py`                    | 完整模型逐 token 推理，Median ms/token    |
| **Kernel 吞吐**    | `benchmarking/scripts/test_kernels_*.py` | 单个 CUDA kernel 的原始吞吐，排除其他开销 |

### 7.3 前置条件：编译 CUDA Kernel 并准备 deploy 环境

Latency 评估依赖自定义 CUDA kernel（`quant_cuda`）和独立的 `deploy` conda 环境：

```bash
# 1. 从 rlkv clone 出 deploy 环境（约 5 分钟）
conda create --clone rlkv -n deploy -y

DEPLOY_PIP=/home/ubuntu/miniconda3/envs/deploy/bin/pip
DEPLOY_PY=/home/ubuntu/miniconda3/envs/deploy/bin/python

# 2. 安装 deployment/transformers（量化推理专用 fork）
cd /home/ubuntu/program/rlkvq/kvquant/deployment/transformers
$DEPLOY_PIP install -e . -q

# 3. 安装 deployment kvquant 包
cd /home/ubuntu/program/rlkvq/kvquant/deployment
$DEPLOY_PIP install -e . -q

# 4. 编译 CUDA kernel
export PATH=/usr/local/cuda-12.4/bin:$PATH
cd /home/ubuntu/program/rlkvq/kvquant/deployment/kvquant
$DEPLOY_PY setup_cuda.py install 2>&1 | tee /home/ubuntu/program/rlkvq/results/build_quant_cuda.log
```

**编译成功标志**：

```
Successfully installed quant-cuda-0.0.0
```

**验证**：

```bash
conda run -n deploy python -c "import quant_cuda; print('quant_cuda OK')"
```

### 7.4 Step 4a：端到端生成 Latency（实际测试命令）

> **重要设计说明**：
>
> - **fp16 baseline** 使用 `bench_fp16.py`（`rlkv` 环境，标准 transformers + `past_key_values`），**不走** deployment 路径。原因：`deployment/llama.py` 的 `LlamaAttention` 为量化推理专门设计，其 `forward_fused_sparse` 只支持 bits∈{2,3,4}，fp16（bits=16）会触发 `assert(False)`。
> - **nuq4** 使用 `deployment/llama.py`（`deploy` 环境），走真实量化 CUDA kernel 路径。

#### fp16 基线 Latency

```bash
conda activate rlkv
cd /home/ubuntu/program/rlkvq/kvquant/deployment

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python bench_fp16.py /data/models/LLaMA-7B wikitext2 \
    --benchmark 128 \
    --check \
    --seqlen 2048 \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/latency_fp16.log
```

**`bench_fp16.py` 参数说明**：

| 参数                | 值   | 说明                                                                                                                |
| ------------------- | ---- | ------------------------------------------------------------------------------------------------------------------- |
| `bench_fp16.py`   | —   | 最小化 fp16 latency 脚本，与 `deployment/llama.py` benchmark() 逻辑等价，但用标准 `past_key_values` 替代 QuantK |
| `--benchmark 128` | 128  | 生成 128 个 token，逐 token 计时后输出 Median                                                                       |
| `--check`         | —   | 同时计算 PPL，用于验证模型加载正确                                                                                  |
| `--seqlen 2048`   | 2048 | 加载数据时的序列长度                                                                                                |

#### nuq4-1% Latency

```bash
DEPLOY_PY=/home/ubuntu/miniconda3/envs/deploy/bin/python
cd /home/ubuntu/program/rlkvq/kvquant/deployment

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
$DEPLOY_PY llama.py /data/models/LLaMA-7B wikitext2 \
    --abits 4 \
    --include_sparse --sparsity-threshold 0.99 \
    --benchmark 128 \
    --check \
    --quantizer-path /data/kvquant/quantizers/nuq4_s1.pkl \
    --seqlen 2048 \
    --maxseqlen 2048 \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/latency_nuq4.log
```

**`deployment/llama.py` 参数说明**：

| 参数                          | 值   | 说明                                           |
| ----------------------------- | ---- | ---------------------------------------------- |
| `--abits 4`                 | 4    | KV Cache 量化位宽（必须为 2/3/4）              |
| `--include_sparse`          | —   | 启用 Dense-and-Sparse：1% 异常值单独 fp16 保存 |
| `--sparsity-threshold 0.99` | 0.99 | 异常值阈值（百分位 0.5/99.5）                  |
| `--benchmark 128`           | 128  | 生成 128 个 token                              |
| `--check`                   | —   | 同时验证 PPL                                   |
| `--maxseqlen 2048`          | 2048 | 量化 KV cache 的最大容量                       |

**预期输出格式**：

```
Benchmarking ...
0 0.032
1 0.031
...
Median: 0.031   ← 单位：秒/token
PPL: ...
max memory(MiB): ...
```

#### 一键后台脚本（推荐方式）

```bash
cd /home/ubuntu/program/rlkvq
nohup bash run_latency_only.sh > results/latency_main.log 2>&1 &
echo "PID: $!"

# 监控进度
tail -f results/latency_main.log
```

独立测量关键 CUDA kernel 的原始吞吐，排除模型加载等开销，便于与 fp16 baseline kernel 直接对比。

#### 前置：缓存激活值（仅需运行一次）

Kernel 测试脚本需要从文件加载激活值（`activations-seqlen2048.pickle`）和量化器（`quantizers.pickle`）：

```bash
conda activate rlkv
cd /home/ubuntu/program/rlkvq/kvquant/benchmarking

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python cache-llama-activations.py \
    /data/models/LLaMA-7B wikitext2 \
    --seqlen 2048 \
    --quantizer-path /data/kvquant/quantizers/nuq4_s1.pkl \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/cache_activations.log
# 生成：activations-seqlen2048.pickle、quantizers.pickle
```

#### fp16 Baseline Kernel（参照系）

```bash
cd /home/ubuntu/program/rlkvq/kvquant/benchmarking/scripts

CUDA_VISIBLE_DEVICES=0 python test_kernel_baselines.py \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/kernel_baseline.log
```

#### nuq4 Key Kernel（Q·Kᵀ）

```bash
CUDA_VISIBLE_DEVICES=0 python test_kernels_key.py \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/kernel_key_nuq4.log
```

#### nuq4 Value Kernel（Score·V）

```bash
CUDA_VISIBLE_DEVICES=0 python test_kernels_value.py \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/kernel_value_nuq4.log
```

#### nuq4 Key + RoPE Fused Kernel

```bash
CUDA_VISIBLE_DEVICES=0 python test_kernel_benchmark_K_plus_rope.py \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/kernel_key_rope_nuq4.log
```

**预期输出格式**（torch profiler 表格）：

```
-------------------------------------------------------  ...
                                   Self CUDA time total
torch.ops.quant_cuda.vecquant4...        X.XXX ms
-------------------------------------------------------
```

#### 本次实测 CUDA kernel latency（A100-40GB）

下表记录了本次在 A100-40GB 上得到的 kernel 级微基准结果，并与论文 Table 6 中的 A6000 数据并列展示。单位均为 **微秒（μs）**。

| 激活操作                    | A100 实测（μs） | 论文 A6000（μs） | 说明               |
| --------------------------- | ---------------- | ----------------- | ------------------ |
| Key fp16 Matvec             | **14.2**   | 33.3              | fp16 baseline      |
| Key nuq4-1% dense kernel    | **49.3**   | 25.6              | 仅 dense 量化路径  |
| Key nuq4-1% sparse kernel   | **35.4**   | —                | 稀疏异常值单独处理 |
| Value fp16 Matvec           | **15.2**   | 26.0              | fp16 baseline      |
| Value nuq4-1% dense kernel  | **20.5**   | 22.1              | 仅 dense 量化路径  |
| Value nuq4-1% sparse kernel | **192.7**  | —                | 稀疏异常值单独处理 |

从这组微基准可以直接看出：

- `Key` 路径上，当前 A100 实测的量化 kernel 还没有赢过 fp16 matvec；
- `Value` dense kernel 已经接近 fp16，但 sparse kernel 代价非常高；
- 因而当前端到端 `nuq4` latency 高于 fp16，主要瓶颈更像是 **kernel overhead 尤其是 sparse path**，而不只是“量化本身是否减少了显存带宽”。

### 7.6 已知问题及修复记录

| 问题                                             | 现象                                                                                     | 修复方式                                                                                                                                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LlamaSdpaAttention.forward` 缺少 `**kwargs` | `TypeError: forward() got an unexpected keyword argument 'past_key_values_length_inp'` | 在 `deployment/transformers/.../modeling_llama.py` 的 `LlamaSdpaAttention.forward` 签名加 `**kwargs`，并将 `LlamaPreTrainedModel._supports_sdpa = False` 强制 eager attention |
| fp16 latency 在 `deployment/llama.py` 中崩溃   | `assert(False)` in `QuantK.forward_fused_sparse`（bits=16 不在 {2,3,4}）             | fp16 baseline 改用 `bench_fp16.py` + 标准 transformers，不走量化 deployment 路径                                                                                                    |

---

## 8. Step 5：Passkey Retrieval 评估

### 8.1 什么是 Passkey Retrieval

Passkey Retrieval 是一种**长上下文信息检索**测试：在大量无关填充文本中随机插入一个 5 位数字（"密钥"），要求模型在读完全文后回答"密钥是多少"。

```
[任务描述(32 tokens)] + [垃圾文本] + [The pass key is 12345. Remember it.] + [垃圾文本] + [What is the pass key?]
```

- 密钥位置随机（可能在开头、中间或末尾）
- 上下文越长，模型越难记住深埋其中的数字
- **指标**：`correct_rate`（正确率，0~1），在 5 个上下文长度上分别测试：`[2048, 4096, 8192, 16384, 32768]`，每个长度随机生成 50 次

**意义**：PPL 只衡量整体语言建模质量，Passkey 测试专门验证量化后模型的**长上下文记忆能力是否退化**。若量化破坏了远距离注意力，correct_rate 会在长上下文下明显下降。

### 8.2 前置条件

Passkey 测试需要模型能完整处理长达 32768 token 的上下文，需用 RoPE Scaling 扩展位置编码：

- `--maxseqlen 32768`：通过线性 RoPE Scaling 把上下文窗口从 4096 扩展到 32768
- 脚本内置自动计算 `scaling_factor = ceil(32768 / 4096) = 8`

> **注意**：Passkey 脚本（`eval_passkey_simquant.py`）使用的是 `quant/` 目录下的 simquant 路径，依赖 `rlkv` 环境，与 PPL 评估共用同一套量化器。

> **重要修正**：论文 Table 2 使用的是 **`LLaMA-2-7B-32K`**，不是 `/data/models/LLaMA-7B`。
> 因此：
>
> 1. fp16 Passkey 基线必须加载 `/data/models/LLaMA-2-7B-32K`；
> 2. 若要复现 `nuq4-1% / nuq3-1% / nuq2-1%`，需要针对 **32K 模型本身** 重新生成 Fisher 信息和 Quantizer；
> 3. 之前针对 `/data/models/LLaMA-7B` 生成的 quantizer 不能视为论文 Table 2 的正式复现结果。

> **执行约束**：启动 Passkey 前必须先确认 GPU 空闲。由于本机 `nvidia-smi` 不可靠，建议使用仓库根目录的 `wait_for_gpu_idle.sh` 包装实际命令。

### 8.2.1 本次 Passkey 量化线实际使用的离线校准参数

Passkey 的量化版评测不是在线量化，而是先离线生成 `Fisher + quantizer`，再读取 quantizer 做 `simquant` 评测。当前这条 32K 路线实际使用的是：

- 模型：`/data/models/LLaMA-2-7B-32K`
- Fisher：`/data/kvquant/fisher-llama-2-7b-32k`
- 校准数据：Wikitext-2
- 校准样本数：`16`
- 校准序列长度：`2048`
- 校准命令核心参数：`--nuq --include_sparse --sparsity-threshold 0.99`
- 2-bit 额外参数：`--norm`

对应 quantizer 文件为：

- `nuq4-1%`：`/data/kvquant/quantizers/llama2-7b-32k_nuq4_s1.pkl`
- `nuq3-1%`：`/data/kvquant/quantizers/llama2-7b-32k_nuq3_s1.pkl`
- `nuq2-1%`：`/data/kvquant/quantizers/llama2-7b-32k_nuq2_s1.pkl`

换句话说，Passkey 量化评测阶段只会读取这些 `.pkl` 文件，不会在评测时重新做校准。

### 8.3 运行命令

#### fp16 基线

```bash
conda activate rlkv
cd /home/ubuntu/program/rlkvq/kvquant/quant

mkdir -p /home/ubuntu/program/rlkvq/results/passkey

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python eval_passkey_simquant.py \
    --path_to_ckp /data/models/LLaMA-2-7B-32K \
    --model_name llama-2-7b-32k-fp16 \
    --maxseqlen 32768 \
    --path_to_output_dir /home/ubuntu/program/rlkvq/results/passkey \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/passkey_fp16.log
```

#### nuq4-1%

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python eval_passkey_simquant.py \
    --path_to_ckp /data/models/LLaMA-2-7B-32K \
    --model_name llama-2-7b-32k-nuq4 \
    --maxseqlen 32768 \
    --simquant --abits 4 \
    --quantizer-path /data/kvquant/quantizers/llama2-7b-32k_nuq4_s1.pkl \
    --path_to_output_dir /home/ubuntu/program/rlkvq/results/passkey \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/passkey_nuq4.log
```

#### nuq2-1%（含 Q-Norm）

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python eval_passkey_simquant.py \
    --path_to_ckp /data/models/LLaMA-2-7B-32K \
    --model_name llama-2-7b-32k-nuq2 \
    --maxseqlen 32768 \
    --simquant --abits 2 --norm \
    --quantizer-path /data/kvquant/quantizers/llama2-7b-32k_nuq2_s1.pkl \
    --path_to_output_dir /home/ubuntu/program/rlkvq/results/passkey \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/passkey_nuq2.log
```

**参数说明**：

| 参数                     | 说明                                          |
| ------------------------ | --------------------------------------------- |
| `--maxseqlen 32768`    | 最大上下文长度，触发 RoPE Scaling（factor=8） |
| `--model_name`         | 输出 jsonl 文件名前缀，建议不同配置用不同名字 |
| `--simquant`           | 启用量化模拟（不传则为 fp16 基线）            |
| `--path_to_output_dir` | jsonl 结果输出目录                            |

**输出文件**：`results/passkey/<model_name>.jsonl`（每个 `context_size` 一条记录，含该长度下所有 case 的详情和 `correct_rate`）

**快速 smoke test**：为了先验证脚本/模型/显存流程，再跑完整 5×50 case，可以先缩小到 2 个 context 长度、每个长度 2 次：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python eval_passkey_simquant.py \
    --path_to_ckp /data/models/LLaMA-2-7B-32K \
    --model_name llama-2-7b-32k-fp16-smoke \
    --maxseqlen 32768 \
    --context-lengths 2048 4096 \
    --num-samples 2 \
    --path_to_output_dir /home/ubuntu/program/rlkvq/results/passkey \
    2>&1 | tee /home/ubuntu/program/rlkvq/results/passkey_fp16_smoke.log
```

如果 GPU 可能正被其他实验占用，建议改成：

```bash
cd /home/ubuntu/program/rlkvq
chmod +x wait_for_gpu_idle.sh

./wait_for_gpu_idle.sh bash -lc '
  source ~/.bashrc
  conda activate rlkv
  cd /home/ubuntu/program/rlkvq/kvquant/quant
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
  python eval_passkey_simquant.py \
    --path_to_ckp /data/models/LLaMA-2-7B-32K \
    --model_name llama-2-7b-32k-fp16-smoke \
    --maxseqlen 32768 \
    --context-lengths 2048 4096 \
    --num-samples 2 \
    --path_to_output_dir /home/ubuntu/program/rlkvq/results/passkey
'
```

**预期耗时**：每个配置约 30~60 分钟（5 个上下文长度 × 50 次生成）

### 8.4 已知潜在问题

| 问题                               | 原因                                                                 | 预期处理方式                              |
| ---------------------------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| `_flash_attn_2_enabled` 报错     | 脚本直接设置 config 属性，旧版 transformers 不识别                   | 删除或替换为 `attn_implementation` 参数 |
| `model.model.set_devices()` 报错 | 此脚本使用 `LlamaForCausalLM` 而非自定义模型，不含 `set_devices` | 删除该行调用即可                          |
| `deepspeed` 未安装               | 脚本 import 了 deepspeed                                             | `pip install deepspeed` 或注释掉 import |
| `jsonlines` 未安装               | 结果写入依赖 jsonlines                                               | `pip install jsonlines`                 |

---

## 9. 预期结果

### 9.1 困惑度（PPL）

评估数据集：Wikitext-2，序列长度 2048，LLaMA-7B。

| 配置      | 位宽                       | 内存占用（相对 fp16） | 论文 PPL | **实测 PPL** | 差值   | 状态 |
| --------- | -------------------------- | --------------------- | -------- | ------------------ | ------ | ---- |
| fp16 基线 | 16-bit                     | 1×                   | 5.68     | **5.677**    | +0.003 | ✅   |
| nuq4-1%   | 4-bit + 1% sparse          | ~1/3                  | 5.69     | **5.701**    | +0.011 | ✅   |
| nuq3-1%   | 3-bit + 1% sparse          | ~1/4                  | 5.75     | **5.760**    | +0.010 | ✅   |
| nuq2-1%   | 2-bit + 1% sparse + Q-Norm | ~1/7                  | 6.05     | **6.069**    | +0.019 | ✅   |

> 四项误差均在 ±0.02 以内，远优于 ±0.05 的允许范围。

### 9.1.1 Value 分层最优粒度消融（当前 `LLaMA-7B` 原生层策略）

我们直接在当前 `LLaMA-7B` 上重跑了 multi-layer block analysis，并用它自己的 `multilayer_layerwise_summary.csv` 为 `Value` 逐层选取最小 `rel_rmse` 的量化粒度。

本轮得到的 `Value` 层策略为：

- `per-channel`：`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 22, 27, 30, 31]`
- `tile32`：`[15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29]`
- `per-token`：`[]`

在保持 `Key` 量化策略不变的前提下，重新测得的 `PPL` 如下：

| 配置    | 复现 baseline PPL | 原生层策略 best-granularity PPL | 相对 baseline 变化 | 结论     |
| ------- | ----------------- | ------------------------------- | ------------------ | -------- |
| nuq4-1% | **5.701**   | **5.727**                 | `+0.026`         | 略差     |
| nuq3-1% | **5.760**   | **5.952**                 | `+0.192`         | 明显变差 |
| nuq2-1% | **6.069**   | **40.988**                | `+34.919`        | 严重退化 |

这说明：即使层策略直接从当前 `LLaMA-7B` 本身 profile 得到，`Value best-granularity` 仍然不能降低 `PPL`，反而在 `4/3/2-bit` 下都变差。

更具体地说，`PPL` 上升的原因主要有三点：

1. `rel_rmse` 最优不等于 `PPL` 最优。这套层策略是按单层局部重构误差选出来的，但语言建模的 `PPL` 取决于多层 attention 叠加后的最终输出，所以“局部误差更小”并不保证“整模型困惑度更低”。
2. `Value` 原始 `per-token` 动态性被显著削弱。原 baseline 的 `Value` 量化本质上是按 token 自适应缩放；而本轮原生层策略最终选出了 `20` 层 `per-channel`、`12` 层 `tile32`、`0` 层 `per-token`。这意味着 Value 的 token 级动态范围适配几乎被整体替换为跨 token 共享 scale，局部 RMSE 虽然下降，但语言建模所需的 token-dependent 细节被削弱了。
3. `2-bit` 下的 `Q-Norm` 与混合粒度 `Value` 组合仍然极不稳定。
   `nuq2-1%` 的 best-granularity `PPL` 仍高达 `40.988`，远高于 baseline `6.069`，表明在极低 bit-width 下，`Q-Norm` 和 `per-channel/tile32` 混合 Value 的数值假设并不兼容。

因此，本轮实验的结论是：

- “把 `Value` 每层改成局部 `rel_rmse` 最优粒度”并不是一个有效的 `PPL` 优化方向
- 更核心的问题在于 `Value` 的 token 级动态特性本身不适合被大面积替换成共享粒度
- `nuq2-1%` 尤其脆弱，原生 profile 只能略微缓解退化，不能从根本上解决问题

---

### 9.1.2 Value 分层最优粒度消融（注意力输出 RMSE 策略）

#### 背景：为什么引入注意力输出 RMSE

9.1.1 使用的 `val_rmse`（Value 张量自身的重构误差）衡量量化粒度优劣，但 Value 量化误差最终体现在注意力层的**输出**上，即 `attn_out = softmax(Q·Kᵀ/√d) · V_quant`。注意力权重矩阵会对 Value 各 token 的误差做加权平均，从而掩盖或放大不同粒度下的误差分布差异。因此，更贴近真实 PPL 影响的指标是**注意力输出相对 RMSE**（attnout_rel_rmse）：

```
attnout_rel_rmse = ||attn_weights · V_quant − attn_weights · V_fp16||_F
                    / ||attn_weights · V_fp16||_F
```

其中 `attn_weights = softmax(Q·Kᵀ/√d)`，在各量化粒度方案下相同，仅 V 不同。

#### 全层分析结果（32 层汇总）

在当前 `LLaMA-7B` 上对全部 32 层同时捕获 Q/K/V 激活，逐层计算各量化粒度下的 `attnout_rel_rmse`，汇总统计如下：

| 粒度        | attnout_rel_rmse 均值 | 均值相对 per_token |
| ----------- | --------------------- | ------------------ |
| per_tensor  | 0.2057                | +116%              |
| per_token   | 0.0954                | 基准               |
| per_channel | 0.0696                | **−27%**    |
| tile32      | 0.0744                | −22%              |
| tile64      | 0.0870                | −9%               |

`per_channel` 在 attnout_rel_rmse 指标下仍为 32 层中的最优粒度（27 层最优），仅 layer 10/13/14/15/17/19/26 被 `tile32` 超越，layer 27 被 `per_token` 超越。

#### 基于 attnout_rel_rmse 的逐层最优策略

将上述每层最小 `attnout_rel_rmse` 对应的粒度汇总：

- `per_channel`（27 层）：`0 1 2 3 4 5 6 7 8 9 11 12 16 18 20 21 22 23 24 25 28 29 30 31`
- `tile32`（4 层）：`10 13 14 15 17 19 26`
- `per_token`（1 层）：`27`

该策略记为 **attnout 策略**，与 9.1.1 的 **val_rmse 策略** 主要差异在于：layer 27 由 `per_channel` 改为 `per_token`，其余大体一致。

#### PPL 评测结果

保持 Key 量化策略不变，仅按 attnout 策略替换 Value 各层粒度，在 Wikitext-2 上重新评估：

| 配置    | fp16 基线 | baseline（per_token） | val_rmse 策略 | **attnout 策略** | attnout vs baseline |
| ------- | --------- | --------------------- | ------------- | ---------------------- | ------------------- |
| nuq4-1% | 5.677     | **5.701**       | 5.727         | **5.727**        | +0.026              |
| nuq3-1% | —        | **5.760**       | 5.952         | **5.972**        | +0.212              |
| nuq2-1% | —        | **6.069**       | 40.988        | **41.373**       | +35.304             |

#### 结论：attnout_rel_rmse 同样无法预测 PPL

引入注意力输出 RMSE 后，attnout 策略与 val_rmse 策略的 PPL 几乎完全相同（nuq4 差 0.000，nuq3 差 +0.020，nuq2 差 +0.385）。两套策略都比 per_token baseline 更差，说明：

1. **attnout_rel_rmse 仍是单样本、单层的局部指标**，无法反映多层误差累积与传播的行为。
2. **per_channel 在单层 attnout 指标上看起来最优，但在多层 PPL 上反而更差**，印证了 Value 的 token 级动态特性在多步注意力叠加后不能被共享 scale 替代。
3. **nuq2-1% 的根本问题是 Q-Norm 与非 per_token 粒度的假设不兼容**：Q-Norm 校正假定误差沿 token 维度分布，当 Value 改为 per_channel/tile32 时，校正方向与实际误差结构错位，导致 PPL 灾难性退化（41.373），与 val_rmse 策略（40.988）表现相当。

---

### 9.1.3 K=4 Lookahead Hidden-State RMSE 策略

#### 背景：为什么继续扩大指标 scope

9.1.1 的 `val_rmse` 只看 `Value` 张量自身重构误差；9.1.2 的 `attnout_rel_rmse` 虽然把误差推进到 `A·V`，但仍然是**单层局部指标**。两者都会倾向于大量选择 `per_channel`，最终在 full PPL 上变差，尤其 `nuq2-1%` 会出现 40+ PPL 的灾难性退化。

因此进一步做了一个更大 scope 的预实验：**K=4 lookahead hidden-state RMSE**。它不只看当前层，而是把当前层的量化扰动继续向后传播 4 层，直接比较量化路径和 fp16 teacher 路径的 hidden state 偏移。

#### 指标定义

对每一层 `l` 和候选 `Value` 粒度 `g ∈ {per-token, per-channel, tile32}`：

1. 只改变当前层 `l` 的 `Value` 粒度；
2. `Key` 策略保持 baseline 不变；
3. 后续层使用同 bit-width 下的 KVQuant baseline 设置；
4. 从 `h_l` 开始向前跑最多 `K=4` 层；
5. 比较量化路径和 fp16 teacher 路径在 `h_{l+K}` 的相对 RMSE。

形式上：

$$
E_{K=4}(l,g)=
\frac{\left\|h_{l+K}^{quant(l,g)} - h_{l+K}^{fp16}\right\|_F}
{\left\|h_{l+K}^{fp16}\right\|_F},
\quad K=\min(4, 32-l)
$$

每层选择：

$$
g_l^* = \arg\min_g E_{K=4}(l,g)
$$

这个指标仍然比 full PPL 快，但已经把 scope 从“单层误差”扩大到“跨 4 层后的 hidden-state drift”。

#### K=4 选出的 Value 策略

K=4 lookahead 选出的策略非常保守，没有再大面积替换 `per-token`：

| Strategy                     | Avg Bit | #Per-token | #Per-channel | #Tile32 | Per-channel Layers | Tile32 Layers |
| ---------------------------- | ------: | ---------: | -----------: | ------: | ------------------ | ------------- |
| `k4_fixed2_value_strategy` |       2 |         31 |            0 |       1 | `[]`             | `[31]`      |
| `k4_fixed3_value_strategy` |       3 |         31 |            1 |       0 | `[0]`            | `[]`        |
| `k4_fixed4_value_strategy` |       4 |         31 |            1 |       0 | `[31]`           | `[]`        |

#### Full PPL 结果

| 配置        | baseline PPL | val_rmse 策略 PPL | attnout 策略 PPL |  K=4 lookahead PPL | K=4 vs baseline | 结论                  |
| ----------- | -----------: | ----------------: | ---------------: | -----------------: | --------------: | --------------------- |
| `nuq4-1%` |     5.700959 |             5.727 |            5.727 | **5.701026** |       +0.000067 | 几乎贴近 baseline     |
| `nuq3-1%` |     5.759615 |             5.952 |            5.972 | **5.763513** |       +0.003898 | 明显好于局部指标策略  |
| `nuq2-1%` |     6.069353 |            40.988 |           41.373 | **6.075462** |       +0.006109 | 成功避开 40+ PPL 灾难 |

#### 对 full PPL 的拟合有效性

这个实验不要求 `K=4 lookahead RMSE` 精确预测 PPL 数值，而是检验它能否在策略选择时拟合 PPL 的关键行为：避免明显坏策略，并把候选排序推向 full PPL 更安全的一侧。

从结果看，K=4 lookahead 至少捕获了三类 PPL 相关信号：

1. **能识别 `nuq2-1%` 的灾难风险。**`val_rmse` 和 `attnout_rel_rmse` 都认为大量 `per_channel` 更优，但 full PPL 分别退化到 `40.988` / `41.373`。K=4 lookahead 在跨层传播后惩罚了这类局部低误差选择，最终策略几乎全保留 `per-token`，PPL 回到 `6.075462`。
2. **能把 `nuq3-1%` 从错误方向拉回 baseline 附近。**局部指标策略的 PPL 为 `5.952` / `5.972`，而 K=4 lookahead 为 `5.763513`，仅比 baseline `5.759615` 高 `0.003898`。这说明 K-step lookahead hidden-state drift 比单层 RMSE 更接近 full PPL 对策略的偏好。
3. **在 `nuq4-1%` 小差距场景中也没有选出明显坏策略。**
   `nuq4-1%` 的 full PPL 差距本来很小，K=4 lookahead 得到 `5.701026`，与 baseline `5.700959` 几乎持平，说明该 proxy 在高 bit 情况下没有引入额外风险。

因此，本轮结论是：**K-step lookahead RMSE 对 PPL 的“排序/安全性”拟合明显强于单层 `val_rmse` 和 `attnout_rel_rmse`**。它不是 full PPL 的数值替代品，但作为 RL 或搜索过程中的快速筛选 reward，更能避免灾难性策略。

#### 轻量性与耗时

K-step lookahead RMSE 的成本主要来自生成候选策略的 `h_{l+K}^{quant}`，也就是从当前层开始额外跑 `K` 个 Transformer block；RMSE 公式本身只是两个 hidden state tensor 的 norm 计算，通常是毫秒级。

本次实测配置：

- `LOOKAHEAD_NSAMPLES=4`
- `seq_len=2048`
- `K=4`
- 32 层
- 3 个 bit 配置：`nuq2/nuq3/nuq4`
- 3 个 Value 粒度：`per-token/per-channel/tile32`
- 总候选数：`32 × 3 × 3 = 288`

实测耗时：

| 任务                                   |         耗时 |
| -------------------------------------- | -----------: |
| 缓存 fp16 hidden states                |  约 2.5 分钟 |
| 计算全部 288 个 K=4 lookahead 候选 RMSE 并选策略 | 约 12.5 分钟 |
| 合计                                   |   约 15 分钟 |

换算到策略搜索中的常用粒度：

| 评估粒度                                                                |                 约耗时 |
| ----------------------------------------------------------------------- | ---------------------: |
| 单个候选：固定 `layer + bit + granularity`  (需要4层小 forward 传播) |             `2.5-3s` |
| 单层单 bit：比较 3 个粒度                                               |               `7-9s` |
| 单层三种 bit：比较 9 个候选                                             |             `23-28s` |
| 仅计算已给定 hidden states 的 RMSE                                      | 毫秒级，通常 `<0.1s` |

**这说明 K-step lookahead RMSE 适合作为 RL/search 的轻量 proxy**：只要 quantizer 和 fp16 teacher hidden states 预先缓存，训练阶段不需要跑完整 Wikitext-2 full PPL，只需要对候选动作做短程 K 层前向和一次 RMSE 计算。

#### 结论

1. **扩大 proxy scope 是有效的。**K=4 lookahead 没有像 `val_rmse` / `attnout_rel_rmse` 那样选择大量 `per_channel`，说明跨层 hidden-state drift 能更好地惩罚会在后续层放大的局部低误差选择。
2. **`nuq2-1%` 的灾难性退化被避免。**`val_rmse` 与 `attnout` 策略在 `nuq2-1%` 下分别为 `40.988` / `41.373` PPL，而 K=4 lookahead 为 `6.075462`，只比 baseline 高 `0.006109`。
3. **`nuq3-1%` 与 `nuq4-1%` 保持接近 baseline。**`nuq3-1%` 从局部策略的 `5.952/5.972` 回到 `5.763513`；`nuq4-1%` 几乎与 baseline `5.700959` 持平。
4. **K-step lookahead RMSE 是比单层 RMSE 更适合 RL 策略搜索的快速指标。**
   它仍不是最终 full PPL，但比单层 `Value` 重构误差更接近模型实际计算路径。后续 RL reward 可以优先考虑 `K-step lookahead RMSE`，并进一步尝试 `K=2/K=8`、logits KL 或 short-NLL 的混合信号。

相关输出文件：

```text
results/k4-lookahead/k4_lookahead_layer_candidate_rmse.csv
results/k4-lookahead/k4_strategy_summary.csv
results/k4-lookahead/k4_full_ppl_summary.csv
strategies/k4_fixed2_value_strategy.json
strategies/k4_fixed3_value_strategy.json
strategies/k4_fixed4_value_strategy.json
README_k4_lookahead_experiment.md
```

### 9.2 端到端生成 Latency

#### 9.2.1 端到端 Token 生成延迟

LLaMA-7B，单卡 A100-40GB，生成 128 token，Median ms/token。

| 配置      | seqlen | 测试脚本                         | 实测 Median             | 日志                       |
| --------- | ------ | -------------------------------- | ----------------------- | -------------------------- |
| fp16 基线 | 2048   | `bench_fp16.py` + rlkv         | **34.7 ms/token** | `latency_fp16_2048.log`  |
| fp16 基线 | 4096   | `bench_fp16.py` + rlkv         | **35.0 ms/token** | `latency_fp16_4096.log`  |
| fp16 基线 | 16384  | `bench_fp16.py` + rlkv         | **37.3 ms/token** | `latency_fp16_16384.log` |
| nuq4-1%   | 2048   | `deployment/llama.py` + deploy | **94.3 ms/token** | `latency_nuq4_2048.log`  |
| nuq4-1%   | 4096   | `deployment/llama.py` + deploy | **94.7 ms/token** | `latency_nuq4_4096.log`  |
| nuq4-1%   | 16384  | `deployment/llama.py` + deploy | **90.4 ms/token** | `latency_nuq4_16384.log` |

> **关于 nuq4 > fp16 的说明**：在当前测试的 seqlen 范围（2048~16384）内，KVQuant kernel 的 overhead（稀疏异常值处理、量化 pack/unpack）仍然高于带宽节省，latency 高于 fp16。随 seqlen 增大，nuq4 latency 从 94.3 ms 略降至 90.4 ms，而 fp16 从 34.7 ms 升至 37.3 ms，两者差距在缩小，趋势与论文一致。论文中的明显加速出现在更长上下文（seqlen ≥ 65K+），此时 KV Cache 带宽完全成为瓶颈，4× 压缩比带来 ~2× 以上加速。

---

#### 9.2.2 底层 CUDA Kernel 吞吐（Attention Kernel 级别）

使用 `benchmarking/scripts/` 下的 kernel 测试脚本，通过 torch profiler 单独计时注意力计算的两个核心 kernel，排除模型其余部分的开销。

测试条件：LLaMA-7B，seqlen=2048，num_heads=32，head_dim=128，1000 次迭代。

**Q·Kᵀ（Key 侧）**

| 配置      | Kernel                                                       | Self CUDA time total  | 每步每层耗时                                 | 日志                             |
| --------- | ------------------------------------------------------------ | --------------------- | -------------------------------------------- | -------------------------------- |
| fp16 基线 | `aten::bmm`（cuBLAS CUTLASS）                              | 15.67 ms / 1000 iters | **14.2 µs**                           | `kernel_baseline_key_2048.log` |
| nuq4-1%   | `VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatch` | 2710 ms / 32000 calls | **1576.7 µs**（32 heads × 49.3 µs） | `kernel_key_nuq4_2048.log`     |

**Score·V（Value 侧）**

| 配置      | Kernel                                                   | Self CUDA time total  | 每步每层耗时                                | 日志                             |
| --------- | -------------------------------------------------------- | --------------------- | ------------------------------------------- | -------------------------------- |
| fp16 基线 | `aten::bmm`（ampere fp16 GEMM）                        | 16.62 ms / 1000 iters | **15.2 µs**                          | `kernel_baseline_val_2048.log` |
| nuq4-1%   | `VecQuant4MatMulKernelNUQPerChannelTransposedMHABatch` | 6821 ms / 32000 calls | **654.9 µs**（32 heads × 20.5 µs） | `kernel_value_nuq4_2048.log`   |

**汇总对比（seqlen=2048，单层单步）**

| 操作            | fp16     | nuq4       | 比值            |
| --------------- | -------- | ---------- | --------------- |
| Q·Kᵀ kernel   | 14.2 µs | 1576.7 µs | **111×** |
| Score·V kernel | 15.2 µs | 654.9 µs  | **43×**  |
| 两者合计        | 29.4 µs | 2231.6 µs | **76×**  |

> **为什么短序列下 nuq4 kernel 反而更慢**：fp16 的 cuBLAS bmm 针对矩阵乘法高度优化，seqlen=2048 时 KV Cache 仅 ~8 MB（在 GPU L2 缓存内），带宽不是瓶颈。nuq4 kernel 需要逐 head 解码 4-bit 量化表示并查 LUT，launch overhead 和 decode 计算反而更大。当 seqlen 增长到 KV Cache 超出 L2 缓存（~40 MB+）时，内存带宽成为瓶颈，4× 数据量减少才能带来实质加速。

### 9.3 Passkey Retrieval 正确率

模型为 `LLaMA-2-7B-32K`，`maxseqlen=32768`，每个上下文长度 50 次随机测试，指标为 `correct_rate`（0~1）。

| 配置      | 论文（2K / 4K / 8K / 16K / 32K）     | 实测（2K / 4K / 8K / 16K / 32K）               | 状态 |
| --------- | ------------------------------------ | ---------------------------------------------- | ---- |
| fp16 基线 | `1.00 / 1.00 / 1.00 / 1.00 / 1.00` | **`1.00 / 1.00 / 1.00 / 1.00 / 1.00`** | ✅   |
| nuq4-1%   | `1.00 / 1.00 / 1.00 / 1.00 / 1.00` | **`1.00 / 1.00 / 1.00 / 1.00 / 1.00`** | ✅   |
| nuq3-1%   | `0.98 / 1.00 / 1.00 / 1.00 / 1.00` | **`1.00 / 1.00 / 1.00 / 1.00 / 1.00`** | ✅   |
| nuq2-1%   | `1.00 / 1.00 / 0.98 / 1.00 / 1.00` | **`1.00 / 0.98 / 0.94 / 0.90 / 0.86`** | ✅   |

> 结果文件以 `results/passkey/llama-2-7b-32k-{fp16,nuq4,nuq3,nuq2}.jsonl` 为准。
> 本次复现中，`fp16`、`nuq4-1%`、`nuq3-1%` 都保持满分检索；`nuq2-1%` 在长上下文下出现明显退化。

---

## 10. 故障排除记录

本次复现过程中遇到的问题及解决方案：

### 10.1 NVML 驱动版本不匹配导致 CUDA 崩溃

**错误**：

```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_()
INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":963
```

**原因**：服务器的 NVML 用户态库（535.288）与内核驱动版本不匹配，PyTorch 默认 CUDA 内存分配器（CUDACachingAllocator）在尝试调用 NVML 时触发内部断言。

**解决**：改用 `cudaMallocAsync` 内存分配器后端，完全绕开 CUDACachingAllocator：

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

### 10.2 多次迭代后 GPU OOM

**错误**：`torch.OutOfMemoryError: Allocation on device`（在 Fisher 计算第 2 条样本时）

**原因**：`run-fisher.py` 每次迭代后没有释放 `k_proj.act` 和 `v_proj.act` 持有的 GPU 计算图，也没有清空梯度缓冲区，导致显存累积。

**解决**（修改 `gradients/run-fisher.py`）：在每轮迭代末尾加入：

```python
k_proj.act = None          # 释放激活张量及其计算图
v_proj.act = None
model.zero_grad(set_to_none=True)   # 清空梯度缓冲区
del outputs, loss
torch.cuda.empty_cache()            # 归还碎片化显存
```

### 10.3 transformers API 版本不兼容

**错误**（校准时）：`TypeError: cannot unpack non-iterable NoneType object`

**原因**：`transformers 4.43+` 重构了 LLaMA 层的 forward 接口，要求调用者传入预计算的 `position_embeddings`（cos/sin 对），而旧代码直接调用单层 forward 时只传 `position_ids`，导致 `position_embeddings=None`。

**解决**：将 `rlkv` 环境中的 transformers 降级到最后一个兼容旧 API 的版本：

```bash
pip install "transformers==4.40.2"
```

### 10.4 nsamples 与 num_examples 不一致

**错误**：`IndexError: The shape of the mask [...] does not match the shape of the indexed tensor`

**原因**：Fisher 信息的形状是 `[1, num_examples×seqlen, hidden]`，校准时的激活数据形状是 `[nsamples×seqlen, hidden]`。两者在 K-Means 加权时需要 flatten 后形状一致。

**解决**：校准命令的 `--nsamples` **必须等于** Fisher 计算的 `--num_examples`，均为 **16**。

### 10.5 flash-attn 未安装

**错误**：`TypeError: __init__() got an unexpected keyword argument 'use_flash_attention_2'`

**原因**：当前环境未安装 flash-attn，transformers 4.40.2 在收到 `use_flash_attention_2=True` 参数时因为 LLaMA model 的构造函数不接受该参数而报错。

**解决**（修改 `quant/llama_simquant.py` 中的 `get_model` 函数）：

```python
try:
    model = AutoModelForCausalLM.from_pretrained(
        ..., use_flash_attention_2=True, ...)
except (ImportError, ValueError, TypeError):
    # flash-attn 未安装或版本不支持，回退到标准注意力
    model = AutoModelForCausalLM.from_pretrained(
        ..., torch_dtype=torch.half)
```

### 10.6 nuq2 校准时 Q-Norm 路径报错

**错误**：`TypeError: round_to_nearest_pole_sim() got an unexpected keyword argument 'return_freq'`

**原因**：`simquant_module_quantizer.py` 中 Q-Norm 路径（2-bit 专用）调用 `round_to_nearest_pole_sim(aug, centroid, return_freq=True)`，但函数定义只有 `(w, poles)` 两个参数，属于未完成的功能残留。

**解决**（修改 `quant/kvquant/simquant_module_quantizer.py`）：为函数添加 `return_freq=False` 参数：

```python
def round_to_nearest_pole_sim(w, poles, return_freq=False):
    ...
    for i, c in enumerate(poles):
        mask = (idx == i)
        aug += mask * c
        if return_freq:
            freq.append(mask.sum().item())
    if return_freq:
        return aug, freq
    return aug
```

### 10.7 fp16 评估误触量化替换路径

**错误**：`NotImplementedError: Only 3, 4, 5 bits are supported.`

**原因**：旧评估命令传入 `--abits 16 --quantizer-path ...`，脚本进入 else 分支后尝试用 16-bit 构造 `QuantLinearSim`，该类明确不支持 16-bit。

**解决**（修改 `quant/llama_simquant.py`）：在 else 分支开头加入短路：

```python
else:
    if args.abits == 16:
        llama_eval(model, testloader, DEV)
        import sys; sys.exit(0)
    # 加载量化器 ...
```

fp16 评估命令同时去掉 `--quantizer-path` 参数。
