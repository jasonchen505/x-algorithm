# X-Algorithm 项目复现 Plan

> 基于 8卡 3090 算力，完整全流程复现 X (Twitter) For You Feed 推荐系统

---

## 目录

- [一、资源评估与可行性分析](#一资源评估与可行性分析)
- [二、复现范围与策略](#二复现范围与策略)
- [三、分阶段执行计划](#三分阶段执行计划)
- [四、环境配置](#四环境配置)
- [五、各阶段详细任务](#五各阶段详细任务)
- [六、验收标准](#六验收标准)
- [七、风险与应对](#七风险与应对)

---

## 一、资源评估与可行性分析

### 1.1 硬件资源

| 资源 | 规格 | 可用量 |
|------|------|--------|
| GPU | RTX 3090 (24GB VRAM, 35.6 TFLOPS FP32, 71 TFLOPS FP16) | 8 卡 |
| 总显存 | 24GB × 8 = **192GB** | - |
| 单卡算力 | ~71 TFLOPS (FP16/BF16) | - |
| 总算力 | ~568 TFLOPS (FP16) | - |

### 1.2 项目计算需求

| 组件 | Mini 模型 (开源版) | 生产推测 |
|------|-------------------|----------|
| **Phoenix 参数量** | ~3.85 亿 (embedding 占 99.7%) | 数十亿 |
| **推理显存/模型** | ~2-3 GB | ~35-40 GB |
| **推理 FLOPS/用户** | ~1.4 GFLOPS | ~330 GFLOPS |
| **训练数据** | 无（需自行构造） | 每天 10B 样本 |
| **GROX LLM** | 远程 API（Grok） | 远程 API |

### 1.3 可行性结论

| 维度 | 评估 | 说明 |
|------|------|------|
| **Mini 模型推理** | ✅ 完全可行 | 单卡 2-3GB，可 8 路并行 |
| **Mini 模型训练** | ✅ 可行 | 单卡足够，多卡可加速 |
| **生产模型推理** | ⚠️ 勉强可行 | 需要模型并行或量化 |
| **生产模型训练** | ❌ 不可行 | 需要数千张 A100 |
| **GROX LLM 分类** | ⚠️ 需替代方案 | 原版依赖 Grok API，需用开源模型替代 |
| **Rust 组件** | ✅ 可编译运行 | 需要 Rust 环境 |
| **端到端 Pipeline** | ✅ 可复现 | 需要构造模拟数据 |

---

## 二、复现范围与策略

### 2.1 复现范围

```
Phase 1: 模型架构复现 (Phoenix)
├── 1.1 Transformer 核心 (grok.py)
├── 1.2 排序模型 (recsys_model.py)
├── 1.3 检索模型 (recsys_retrieval_model.py)
└── 1.4 单元测试验证

Phase 2: 推理 Pipeline 复现
├── 2.1 端到端推理 (run_pipeline.py)
├── 2.2 模拟数据构造
└── 2.3 推理性能 benchmark

Phase 3: 训练 Pipeline 构建
├── 3.1 训练数据构造
├── 3.2 训练循环实现
├── 3.3 Mini 模型训练
└── 3.4 训练效果验证

Phase 4: GROX 内容理解复现
├── 4.1 分类器框架复现
├── 4.2 开源 LLM 替代方案
├── 4.3 嵌入模型替代方案
└── 4.4 端到端内容理解测试

Phase 5: 系统集成与端到端测试
├── 5.1 Home Mixer 编译运行
├── 5.2 Thunder 编译运行
├── 5.3 端到端推荐测试
└── 5.4 性能 benchmark
```

### 2.2 复现策略

| 策略 | 说明 |
|------|------|
| **模型架构** | 完整复现，逐行理解代码 |
| **推理 Pipeline** | 完整复现，用模拟数据验证 |
| **训练 Pipeline** | 自行构建，参考 JAX 最佳实践 |
| **GROX LLM** | 用开源模型替代（Qwen2.5-VL / InternVL2） |
| **Rust 组件** | 编译运行，理解架构设计 |
| **外部服务** | Mock 替代（Redis、Kafka、Strato） |

---

## 三、分阶段执行计划

### 总时间线：8-10 周

```
Week 1-2:   Phase 1 - 模型架构复现
Week 3-4:   Phase 2 - 推理 Pipeline 复现
Week 5-6:   Phase 3 - 训练 Pipeline 构建
Week 7-8:   Phase 4 - GROX 内容理解复现
Week 9-10:  Phase 5 - 系统集成与端到端测试
```

---

## 四、环境配置

### 4.1 Python 环境

```bash
# 创建虚拟环境
conda create -n x-algo python=3.11
conda activate x-algo

# 安装 JAX (CUDA 11.8)
pip install jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装依赖
pip install dm-haiku numpy scipy matplotlib tqdm
pip install pytest  # 单元测试

# 安装 GROX 依赖
pip install litellm pydantic kafka-python redis
pip install opentelemetry-api opentelemetry-sdk
```

### 4.2 Rust 环境

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 编译 Home Mixer
cd home-mixer
cargo build --release

# 编译 Thunder
cd ../thunder
cargo build --release
```

### 4.3 数据准备

```bash
# 下载预训练模型（通过 Git LFS）
git lfs install
git lfs pull

# 预期产物：
# phoenix/artifacts/retrieval/model_params.npz (~3MB)
# phoenix/artifacts/retrieval/embedding_tables.npz (~1.4GB)
# phoenix/artifacts/ranker/model_params.npz (~3MB)
# phoenix/artifacts/ranker/embedding_tables.npz (~1.4GB)
# phoenix/artifacts/sports_corpus.npz (~260MB)
# phoenix/artifacts/example_sequence.json (~几KB)
```

---

## 五、各阶段详细任务

### Phase 1: 模型架构复现 (Week 1-2)

#### 1.1 Transformer 核心 (grok.py)

**目标**：深入理解并验证 Grok-based Transformer 的每个组件

**任务清单**：

| 序号 | 任务 | 文件 | 预计时间 |
|------|------|------|----------|
| 1.1.1 | 阅读并理解 `TransformerConfig` | `grok.py:112-123` | 2h |
| 1.1.2 | 阅读并理解 `MultiHeadAttention` | `grok.py:200-400` | 4h |
| 1.1.3 | 阅读并理解 `DenseBlock` (SwiGLU FFN) | `grok.py:400-500` | 2h |
| 1.1.4 | 阅读并理解 `DecoderLayer` (Pre-Norm) | `grok.py:500-530` | 2h |
| 1.1.5 | 阅读并理解 `make_recsys_attn_mask` | `grok.py:39-71` | 3h |
| 1.1.6 | 阅读并理解 `right_anchored_rope_positions` | `grok.py:88-109` | 2h |
| 1.1.7 | 阅读并理解 `RotaryEmbedding` | `grok.py:229-285` | 3h |
| 1.1.8 | 手写 Transformer 单元测试 | 新文件 | 4h |
| 1.1.9 | 可视化 attention mask | 新文件 | 2h |

**验收标准**：
- [ ] 能画出完整的 Transformer 架构图
- [ ] 能解释 Candidate Isolation 的设计动机和实现细节
- [ ] 能解释 Right-Anchored RoPE 的作用
- [ ] 能解释 tanh clipping 的作用
- [ ] 单元测试全部通过

**关键学习点**：
```python
# 1. Candidate Isolation Mask 结构
#              User  History    Candidates
# User       [  ✓ ]
# History    [  ✓      ✓       ]
# Candidate  [  ✓      ✓       diag_only ]

# 2. tanh clipping: 30 * tanh(x/30)
# - 平滑软裁剪，避免梯度不连续
# - 配合 attn_output_multiplier=0.125 使用

# 3. SwiGLU FFN:
# output = Linear(GELU(Linear(x)) * Linear(x))
# ffn_size = int(widening_factor * emb_size) * 2 // 3
# 对齐到 8 的倍数以优化 GPU SIMD
```

#### 1.2 排序模型 (recsys_model.py)

**目标**：理解 Phoenix 排序模型的完整架构

**任务清单**：

| 序号 | 任务 | 文件 | 预计时间 |
|------|------|------|----------|
| 1.2.1 | 阅读并理解 `HashConfig` 和哈希机制 | `recsys_model.py:93-101` | 2h |
| 1.2.2 | 阅读并理解 `block_user_reduce` | `recsys_model.py:147-197` | 3h |
| 1.2.3 | 阅读并理解 `block_history_reduce` | `recsys_model.py:200-268` | 3h |
| 1.2.4 | 阅读并理解 `block_candidate_reduce` | `recsys_model.py:271-332` | 3h |
| 1.2.5 | 阅读并理解 `PhoenixModel.__call__` | `recsys_model.py:628-680` | 4h |
| 1.2.6 | 阅读并理解 Multi-Action Prediction | `runners.py:233-253` | 2h |
| 1.2.7 | 运行并验证 `test_recsys_model.py` | 测试文件 | 2h |
| 1.2.8 | 参数量统计和显存估算 | 新文件 | 2h |

**验收标准**：
- [ ] 能画出排序模型的完整数据流图
- [ ] 能解释 Hash-Based Embedding 的优势和劣势
- [ ] 能解释 19 种 Action 的设计动机
- [ ] 能计算模型参数量和显存占用
- [ ] 单元测试全部通过

#### 1.3 检索模型 (recsys_retrieval_model.py)

**目标**：理解 Two-Tower 检索模型

**任务清单**：

| 序号 | 任务 | 文件 | 预计时间 |
|------|------|------|----------|
| 1.3.1 | 阅读并理解 `CandidateTower` | `recsys_retrieval_model.py:47-112` | 2h |
| 1.3.2 | 阅读并理解 `PhoenixRetrievalModel` | `recsys_retrieval_model.py:150-300` | 3h |
| 1.3.3 | 阅读并理解 User Tower 构建 | `recsys_retrieval_model.py:221-291` | 3h |
| 1.3.4 | 阅读并理解 Top-K 检索 | `recsys_retrieval_model.py:362-387` | 2h |
| 1.3.5 | 运行并验证 `test_recsys_retrieval_model.py` | 测试文件 | 2h |

**验收标准**：
- [ ] 能画出 Two-Tower 架构图
- [ ] 能解释为什么 User Tower 用 Transformer 而 Candidate Tower 用 MLP
- [ ] 能解释 L2 归一化的作用
- [ ] 单元测试全部通过

---

### Phase 2: 推理 Pipeline 复现 (Week 3-4)

#### 2.1 端到端推理 (run_pipeline.py)

**目标**：复现完整的 Retrieval → Ranking Pipeline

**任务清单**：

| 序号 | 任务 | 文件 | 预计时间 |
|------|------|------|----------|
| 2.1.1 | 阅读并理解 `run_pipeline.py` | `run_pipeline.py` | 4h |
| 2.1.2 | 下载预训练模型 artifacts | Git LFS | 1h |
| 2.1.3 | 运行原始 Pipeline（sports 语料库） | `run_pipeline.py` | 2h |
| 2.1.4 | 理解 `_hash_ids` 线性同余哈希 | `run_pipeline.py:76-90` | 2h |
| 2.1.5 | 理解 `build_unified_emb_table` | `run_pipeline.py:100-150` | 2h |
| 2.1.6 | 理解加权排序公式 | `run_pipeline.py:355-361` | 1h |

**验收标准**：
- [ ] 成功运行 `run_pipeline.py`，输出排序结果
- [ ] 理解从用户序列到最终排序的完整流程
- [ ] 能解释每个阶段的数据流和维度变化

#### 2.2 模拟数据构造

**目标**：构造多样化的测试数据，覆盖各种场景

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 2.2.1 | 构造多用户序列（不同长度、不同兴趣） | 4h |
| 2.2.2 | 构造模拟语料库（不同规模：1K/10K/100K） | 4h |
| 2.2.3 | 构造边界测试数据（空序列、超长序列、新用户） | 2h |
| 2.2.4 | 数据格式验证和可视化 | 2h |

**数据格式**：
```json
{
  "user_id": 12345,
  "engagement_history": [
    {
      "post_id": 1001,
      "author_id": 2001,
      "actions": {"0": 1.0, "10": 0.8},
      "product_surface": 1,
      "timestamp": 1719500000
    }
  ]
}
```

#### 2.3 推理性能 Benchmark

**目标**：评估 3090 上的推理性能

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 2.3.1 | 单用户推理延迟测试 | 2h |
| 2.3.2 | 批量推理吞吐量测试 | 2h |
| 2.3.3 | 不同模型规模对比（Mini vs 扩展） | 3h |
| 2.3.4 | 显存占用 profiling | 2h |
| 2.3.5 | 多卡并行推理测试 | 2h |

**Benchmark 指标**：
- 单用户推理延迟 (P50/P99)
- 批量推理吞吐量 (users/sec)
- 显存占用 (GB)
- GPU 利用率 (%)

---

### Phase 3: 训练 Pipeline 构建 (Week 5-6)

#### 3.1 训练数据构造

**目标**：基于现有推理代码，构建训练数据管道

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 3.1.1 | 设计训练数据格式（基于 RecsysBatch） | 2h |
| 3.1.2 | 构造正负样本生成逻辑 | 4h |
| 3.1.3 | 实现数据加载器（JAX/Dataset） | 6h |
| 3.1.4 | 数据增强策略（负采样、序列截断） | 4h |
| 3.1.5 | 数据验证和可视化 | 2h |

**训练数据格式**：
```python
class TrainingExample(NamedTuple):
    user_hashes: Array           # [num_user_hashes]
    history_post_hashes: Array   # [history_len, num_item_hashes]
    history_author_hashes: Array # [history_len, num_author_hashes]
    history_actions: Array       # [history_len, num_actions]
    candidate_post_hashes: Array # [num_candidates, num_item_hashes]
    candidate_author_hashes: Array
    labels: Array                # [num_candidates, num_actions] - 二值标签
    continuous_labels: Array     # [num_candidates, num_continuous_actions]
```

#### 3.2 训练循环实现

**目标**：实现完整的 JAX 训练循环

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 3.2.1 | 实现损失函数（Binary Cross-Entropy + MAE） | 4h |
| 3.2.2 | 实现训练步骤（jax.grad + optax） | 4h |
| 3.2.3 | 实现评估步骤（AUC、NDCG） | 4h |
| 3.2.4 | 实现训练循环（epoch、checkpoint） | 4h |
| 3.2.5 | 实现学习率调度 | 2h |
| 3.2.6 | 实现梯度裁剪和监控 | 2h |

**训练配置**：
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # 多任务权重
    action_weights: dict = field(default_factory=lambda: {
        "favorite": 1.0,
        "reply": 0.5,
        "repost": 0.3,
        "dwell": 0.2,
        "not_interested": -1.0,
        "block_author": -2.0,
    })
```

#### 3.3 Mini 模型训练

**目标**：在 3090 上训练 Mini 模型

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 3.3.1 | 单卡训练验证 | 4h |
| 3.3.2 | 多卡数据并行训练 | 6h |
| 3.3.3 | 训练曲线监控和调优 | 8h |
| 3.3.4 | 模型保存和加载 | 2h |
| 3.3.5 | 推理验证（训练后模型） | 2h |

**训练资源估算**：
- 单卡 3090：Mini 模型训练约 2-4 小时（10 epochs，100K 样本）
- 8 卡并行：约 30-60 分钟

#### 3.4 训练效果验证

**目标**：验证训练后模型的效果

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 3.4.1 | 离线评估指标（AUC、NDCG） | 2h |
| 3.4.2 | 排序质量分析（加权分数分布） | 2h |
| 3.4.3 | 与预训练模型对比 | 2h |
| 3.4.4 | 消融实验（不同超参数） | 4h |

---

### Phase 4: GROX 内容理解复现 (Week 7-8)

#### 4.1 分类器框架复现

**目标**：复现 GROX 的分类器框架

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 4.1.1 | 阅读并理解 `ContentClassifier` 基类 | 2h |
| 4.1.2 | 阅读并理解各分类器实现 | 4h |
| 4.1.3 | 阅读并理解 Plan/DAG 执行引擎 | 4h |
| 4.1.4 | 阅读并理解 Dispatcher/Engine 架构 | 3h |

#### 4.2 开源 LLM 替代方案

**目标**：用开源模型替代 Grok API

**替代方案**：

| 原始模型 | 替代方案 | 模型大小 | 显存需求 |
|---------|---------|---------|---------|
| VLM_PRIMARY | Qwen2.5-VL-7B 或 InternVL2-8B | 7-8B | ~14-16GB |
| VLM_MINI | Qwen2.5-3B-Instruct | 3B | ~6GB |
| VLM_SAFETY | ShieldGemma-2B 或微调模型 | 2B | ~4GB |
| EMBED_PRIMARY | BGE-M3 或 E5-Mistral | 568M / 7B | ~2GB / 14GB |

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 4.2.1 | 部署 Qwen2.5-VL-7B（单卡 3090） | 4h |
| 4.2.2 | 实现 OpenAI 兼容 API 服务 | 4h |
| 4.2.3 | 适配 GROX 分类器调用 | 4h |
| 4.2.4 | Prompt 迁移和验证 | 4h |
| 4.2.5 | 效果对比（Grok vs Qwen） | 4h |

**部署代码**：
```bash
# 使用 vLLM 部署 Qwen2.5-VL-7B
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000
```

#### 4.3 嵌入模型替代方案

**目标**：用开源嵌入模型替代 xAI 嵌入服务

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 4.3.1 | 部署 BGE-M3 嵌入模型 | 2h |
| 4.3.2 | 实现嵌入 API 服务 | 2h |
| 4.3.3 | 适配 GROX 嵌入器调用 | 4h |
| 4.3.4 | 嵌入质量验证 | 2h |

#### 4.4 端到端内容理解测试

**目标**：验证完整的内容理解流程

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 4.4.1 | 构造测试帖子数据 | 2h |
| 4.4.2 | 运行分类流程（垃圾检测、质量评分） | 4h |
| 4.4.3 | 运行嵌入流程 | 2h |
| 4.4.4 | 结果分析和可视化 | 2h |

---

### Phase 5: 系统集成与端到端测试 (Week 9-10)

#### 5.1 Home Mixer 编译运行

**目标**：编译并运行 Home Mixer 服务

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 5.1.1 | 安装 Rust 和依赖 | 1h |
| 5.1.2 | 编译 Home Mixer | 2h |
| 5.1.3 | Mock 外部服务（Redis、Kafka、Strato） | 8h |
| 5.1.4 | 运行 gRPC 服务 | 2h |
| 5.1.5 | 发送测试请求 | 2h |

#### 5.2 Thunder 编译运行

**目标**：编译并运行 Thunder 服务

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 5.2.1 | 编译 Thunder | 1h |
| 5.2.2 | Mock Kafka 消费者 | 4h |
| 5.2.3 | 构造测试帖子数据 | 2h |
| 5.2.4 | 运行 gRPC 服务 | 1h |
| 5.2.5 | 测试帖子检索 | 2h |

#### 5.3 端到端推荐测试

**目标**：验证完整的推荐流程

**任务清单**：

| 序号 | 任务 | 预计时间 |
|------|------|----------|
| 5.3.1 | 构造完整的用户+帖子数据集 | 4h |
| 5.3.2 | 运行端到端推荐流程 | 4h |
| 5.3.3 | 分析推荐结果质量 | 4h |
| 5.3.4 | 性能 profiling 和优化 | 4h |

#### 5.4 性能 Benchmark

**目标**：评估系统整体性能

**Benchmark 指标**：

| 指标 | 目标值 |
|------|--------|
| 单用户推荐延迟 (P50) | < 100ms |
| 单用户推荐延迟 (P99) | < 500ms |
| 推荐吞吐量 | > 100 users/sec |
| 显存占用 | < 24GB (单卡) |
| 推荐多样性 | > 0.7 (作者多样性指数) |

---

## 六、验收标准

### 6.1 代码级验收

| 验收项 | 标准 |
|--------|------|
| 单元测试 | `pytest test_*.py` 全部通过 |
| 代码理解 | 能解释每个关键函数的作用和设计动机 |
| 推理验证 | 成功运行 `run_pipeline.py` 并输出正确结果 |
| 训练验证 | 训练 loss 收敛，评估指标合理 |

### 6.2 系统级验收

| 验收项 | 标准 |
|--------|------|
| 端到端流程 | 从用户输入到推荐输出完整运行 |
| 多组件集成 | Phoenix + GROX + Home Mixer + Thunder 协同工作 |
| 性能指标 | 满足上述 Benchmark 目标 |
| 文档完整 | 每个阶段有详细的学习笔记 |

### 6.3 学习成果验收

| 验收项 | 标准 |
|--------|------|
| 架构理解 | 能画出完整的系统架构图 |
| 原理理解 | 能解释 Candidate Isolation、Hash Embedding、Two-Tower 等核心设计 |
| 工程理解 | 能解释容错、缓存、监控等工程实践 |
| 业务理解 | 能解释推荐系统的业务价值和优化方向 |

---

## 七、风险与应对

### 7.1 技术风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| JAX 在 3090 上兼容性问题 | 无法运行模型 | 使用 PyTorch 重写关键组件 |
| 预训练模型下载失败 | 无法运行推理 | 自行训练 Mini 模型 |
| Rust 编译依赖缺失 | 无法编译 Home Mixer | 使用 Docker 容器编译 |
| 开源 LLM 效果不佳 | GROX 复现质量低 | 使用规则引擎降级 |

### 7.2 资源风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 显存不足 | 无法运行大模型 | 使用量化（INT8/INT4）或模型并行 |
| 训练时间过长 | 超出计划时间 | 减少数据量、简化模型 |
| 存储不足 | 无法保存模型和数据 | 使用外部存储或云服务 |

### 7.3 进度风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 某阶段耗时过长 | 延期 | 优先完成核心功能，非核心功能简化 |
| 遇到难以解决的问题 | 阻塞 | 及时求助、调整计划 |

---

## 附录：每日学习计划模板

```markdown
## Day X (YYYY-MM-DD)

### 今日目标
- [ ] 任务 1
- [ ] 任务 2

### 学习内容
- 关键概念：
- 代码位置：
- 设计决策：

### 遇到的问题
- 问题 1：
  - 原因：
  - 解决方案：

### 今日收获
- 收获 1：
- 收获 2：

### 明日计划
- 计划 1：
- 计划 2：
```

---

*最后更新：2026-06-27*
*基于 8卡 3090 算力评估*
