# LLM 算法实习面试准备指南

> 基于 X (Twitter) For You Feed 推荐系统项目，聚焦 LLM & Agent & GR 应用及后训练相关考察点

---

## 目录

- [第一部分：项目全景与技术栈](#第一部分项目全景与技术栈)
- [第二部分：LLM 在推荐系统中的应用（GROX 内容理解）](#第二部分llm-在推荐系统中的应用grox-内容理解)
- [第三部分：Transformer 在推荐排序中的应用（Phoenix）](#第三部分transformer-在推荐排序中的应用phoenix)
- [第四部分：系统设计与工程实践](#第四部分系统设计与工程实践)
- [第五部分：面试深挖点与考察细节](#第五部分面试深挖点与考察细节)
- [第六部分：候选人项目介绍模板](#第六部分候选人项目介绍模板)
- [第七部分：LLM 通用面试题补充](#第七部分llm-通用面试题补充)

---

## 第一部分：项目全景与技术栈

### 1.1 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FOR YOU FEED REQUEST                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           HOME MIXER                              │
│                      (Rust 编排层, gRPC)                          │
├─────────────────────────────────────────────────────────────────┤
│  Query Hydrators → Sources → Hydrators → Filters → Scorers →    │
│  Selector → Post-Selection → Side Effects                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   THUNDER    │    │   PHOENIX    │    │    GROX      │
│  (Rust)      │    │  (JAX)       │    │  (Python)    │
│  内网内容存储 │    │  检索+排序    │    │  内容理解     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 1.2 技术栈

| 组件 | 语言/框架 | 核心技术 |
|------|----------|---------|
| Home Mixer | Rust | gRPC, Tokio, Candidate Pipeline |
| Thunder | Rust | DashMap, Kafka, gRPC |
| Phoenix | Python/JAX | dm-haiku, Transformer, Two-Tower |
| GROX | Python | LLM/VLM, Kafka, asyncio |

### 1.3 关键设计决策

1. **无手工特征**：完全依赖 Transformer 从用户行为序列学习相关性
2. **Candidate Isolation**：排序时候选之间不相互 attend，保证得分独立性
3. **Hash-Based Embedding**：多哈希函数减少冲突，支持超大 ID 空间
4. **Multi-Action Prediction**：预测 19 种交互概率而非单一相关性分数

---

## 第二部分：LLM 在推荐系统中的应用（GROX 内容理解）

### 2.1 GROX 架构理解

**核心问题**：GROX 如何利用 LLM/VLM 进行内容理解？

```
Kafka 事件流
     │
     ▼
┌─────────────┐
│ Dispatcher  │ ← 任务调度，优先级队列
└─────────────┘
     │
     ▼
┌─────────────┐
│   Engine    │ ← 任务执行引擎
└─────────────┘
     │
     ▼
┌─────────────┐
│ PlanMaster  │ ← 9 个 Plan，DAG 执行
└─────────────┘
     │
     ├── PlanSpamComment (垃圾检测)
     ├── PlanBangerScreen (质量评分)
     ├── PlanPostSafety (安全审查)
     ├── PlanReplyRanking (回复排序)
     └── PlanPostEmbedding (嵌入生成)
```

### 2.2 LLM 分类器设计模式

**考察点 1：ContentClassifier 模板方法**

```python
class ContentClassifier(ABC):
    def classify(self, post):
        # 模板方法：_to_convo → _sample → _parse
        convo = self._to_convo(post)      # 构建对话
        output = self._sample(convo)       # LLM 采样
        return self._parse(post, output)   # 解析输出
```

**面试深挖问题**：
1. 为什么用模板方法模式而不是策略模式？
2. `_to_convo` 如何处理多模态输入（图片、视频）？
3. `_sample` 中 temperature 设置为 0.000001 的原因？

**考察点 2：多级分类体系**

```python
# 安全检测的两阶段设计
1. SafetyPtosCategoryClassifier  → 粗粒度分类（是否有违规）
2. SafetyPtosPolicyClassifier    → 细粒度政策判定（具体违规类型）
```

**面试深挖问题**：
1. 为什么要分两阶段而不是直接细粒度分类？
2. 如何处理类别不平衡（违规样本稀少）？
3. 如何保证分类的一致性和可解释性？

### 2.3 Prompt Engineering 实践

**考察点 3：结构化输出解析**

```python
class SpamSampleResult(BaseModel):
    is_spam: bool
    reason: str

def _parse(self, post, output):
    result = SpamSampleResult.model_validate_json(output)
    return ContentCategory(
        type=ContentCategoryType.SPAM_COMMENT,
        value=result.is_spam,
        metadata={"reason": result.reason}
    )
```

**面试深挖问题**：
1. 如何处理 LLM 输出格式不一致的情况？
2. 为什么用 Pydantic 而不是正则解析？
3. 如何设计 fallback 机制？

**考察点 4：Few-shot Prompt 设计**

从代码中可以看到 prompt 文件的使用：
```python
# 读取 prompt 模板
prompt = self._load_prompt("spam_detection.txt")
# 构建对话
convo = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": self._render_post(post)}
]
```

**面试深挖问题**：
1. 如何设计有效的 few-shot examples？
2. 如何平衡 prompt 长度和效果？
3. 如何进行 prompt 版本管理和 A/B 测试？

### 2.4 多模态嵌入生成

**考察点 5：MultimodalPostEmbedder**

```python
class MultimodalPostEmbedderV5:
    def embed(self, post, transcript, is_query):
        # 1. 渲染帖子（文本 + 图片 + 视频帧）
        rendered = self.renderer.render(post)
        # 2. 拼接 ASR 转录
        if transcript:
            rendered.text += f"\n{transcript}"
        # 3. 调用嵌入 API
        embedding = self.client.encode(rendered)
        # 4. 截断 + 归一化
        return self._maybe_truncate(embedding)
```

**面试深挖问题**：
1. 为什么要截断到 1024 维？权衡什么？
2. 如何处理不同模态的对齐问题？
3. 嵌入模型如何更新？如何保证新旧嵌入兼容？

### 2.5 LLM 应用的工程挑战

**考察点 6：异步 DAG 执行**

```python
class Plan:
    TASK_DEPENDENCIES = {
        "media": [],
        "classifier": ["media"],
        "embedding": ["media"],
        "sink": ["classifier", "embedding"]
    }
    
    async def execute(self, task):
        futures = {}
        for task_name, deps in self.TASK_DEPENDENCIES.items():
            dep_futures = [futures[d] for d in deps]
            futures[task_name] = asyncio.create_task(
                self._execute_task(task_name, dep_futures)
            )
        return await asyncio.gather(*futures.values())
```

**面试深挖问题**：
1. 如何处理 DAG 中的失败节点？
2. 如何实现超时控制？
3. 如何监控和调试复杂的 DAG？

**考察点 7：速率限制与优先级队列**

```python
class PriorityTaskGenerator:
    def _poll(self):
        # 加权随机选择 generator
        weights = [g.weight for g in self.generators]
        selected = random.choices(self.generators, weights=weights)[0]
        return await selected.poll()
```

**面试深挖问题**：
1. 如何设计公平的优先级调度？
2. 如何处理突发流量？
3. 如何保证高优先级任务的延迟 SLA？

---

## 第三部分：Transformer 在推荐排序中的应用（Phoenix）

### 3.1 模型架构

**考察点 8：Grok-based Transformer**

```python
class TransformerConfig:
    emb_size: int = 128          # 嵌入维度
    key_size: int = 32           # 注意力头维度
    num_q_heads: int = 4         # Query 头数
    num_kv_heads: int = 4        # KV 头数（支持 GQA）
    num_layers: int = 4          # 层数
    widening_factor: float = 2.0 # FFN 宽度倍数
```

**面试深挖问题**：
1. 为什么用 128 维而不是更大的模型？权衡什么？
2. widening_factor=2.0 时 FFN 实际维度如何计算？
3. GQA (Grouped Query Attention) 的优势是什么？

**考察点 9：Attention 实现细节**

```python
# tanh clipping 稳定训练
attn_logits = 30.0 * tanh(attn_logits / 30.0)

# Pre-Norm + 双重 Norm
h = inputs
attn_out = MHABlock(...)(RMSNorm(h), mask, positions)
h += RMSNorm(attn_out)  # 注意：输出也 Norm
```

**面试深挖问题**：
1. 为什么用 tanh clipping 而不是其他方法（如 logit soft-capping）？
2. Pre-Norm vs Post-Norm 的优劣？
3. 双重 Norm 的作用是什么？

### 3.2 Candidate Isolation（核心创新）

**考察点 10：自定义 Attention Mask**

```python
def make_recsys_attn_mask(seq_len, candidate_start_offset, dtype):
    # 标准因果 mask
    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))
    
    # 清零 candidate-to-candidate
    attn_mask[:, :, offset:, offset:] = 0
    
    # 恢复 candidate 自注意力
    attn_mask[:, :, indices, indices] = 1
```

生成的 mask 结构：
```
              User  History    Candidates
User       [  ✓ ]
History    [  ✓      ✓       ]
Candidate  [  ✓      ✓       diag_only ]
```

**面试深挖问题**：
1. 为什么需要 Candidate Isolation？
2. 不做 Candidate Isolation 会有什么问题？
3. 这种设计对训练和推理有什么影响？

**核心答案**：
- **独立性保证**：每个候选的得分不依赖于 batch 中有哪些其他候选
- **可缓存性**：得分可以独立缓存，不会因为候选集合变化而失效
- **一致性**：相同用户+相同候选 = 相同得分，无论排序如何

### 3.3 Two-Tower 检索模型

**考察点 11：架构设计**

```
        User Tower                    Candidate Tower
        ┌──────────┐                  ┌──────────────┐
        │  User +   │                 │  Post + Author│
        │  History  │                 │  Embeddings   │
        │  ↓        │                 │  ↓            │
        │ Transformer│                │ MLP (2-layer) │
        │  ↓        │                 │  ↓            │
        │ Mean Pool  │                │ L2 Normalize  │
        │  ↓        │                 │  ↓            │
        │ L2 Norm   │                 │  [B, C, D]    │
        │  ↓        │                 │               │
        │ [B, D]    │                 └──────────────┘
        └──────────┘
```

**面试深挖问题**：
1. 为什么 User Tower 用 Transformer 而 Candidate Tower 用 MLP？
2. Mean Pooling vs CLS Token vs Attention Pooling 的选择？
3. L2 归一化的作用是什么？为什么点积等价于余弦相似度？

**考察点 12：离线 vs 在线计算**

```python
# Candidate Tower 可以离线预计算
corpus_embeddings = candidate_tower.encode_all(corpus)  # 离线
# User Tower 在线计算
user_repr = user_tower.encode(user_history)  # 在线
# 检索 = 点积
scores = user_repr @ corpus_embeddings.T
```

**面试深挖问题**：
1. 为什么要分离 User Tower 和 Candidate Tower？
2. 如何更新 corpus_embeddings？增量更新 vs 全量更新？
3. 如何处理新用户/新内容的冷启动？

### 3.4 Hash-Based Embedding

**考察点 13：多哈希函数**

```python
@dataclass
class HashConfig:
    num_user_hashes: 2     # 每个 user 用 2 个 hash
    num_item_hashes: 2     # 每个 item 用 2 个 hash
    num_author_hashes: 2   # 每个 author 用 2 个 hash
```

**面试深挖问题**：
1. 为什么用多个哈希函数而不是一个大表？
2. 哈希冲突如何处理？
3. 与 Full Embedding Table 相比的优劣？

**核心答案**：
- **内存效率**：无需维护完整 ID→embedding 映射（可能数十亿）
- **冲突缓解**：多个 hash 互相补充，减少单点冲突影响
- **可扩展**：新 ID 无需显式分配 embedding，自动 hash 映射

### 3.5 Multi-Action Prediction

**考察点 14：19 种动作预测**

```python
ACTIONS = [
    "favorite_score", "reply_score", "repost_score", "photo_expand_score",
    "click_score", "profile_click_score", "vqv_score", "share_score",
    "share_via_dm_score", "share_via_copy_link_score", "dwell_score",
    "quote_score", "quoted_click_score", "follow_author_score",
    "not_interested_score", "block_author_score", "mute_author_score",
    "report_score", "dwell_time"
]

# 加权组合
weighted_score = Σ(weight_i × P(action_i))
```

**面试深挖问题**：
1. 为什么预测多个动作而不是单一相关性分数？
2. 如何确定各动作的权重？
3. 正向动作和负向动作如何平衡？

### 3.6 Right-Anchored RoPE

**考察点 15：位置编码创新**

```python
def right_anchored_rope_positions(padding_mask, history_seq_len, num_user_prefix_tokens):
    # 最新的 history token 总是获得固定 position
    # 解决 history 长度不一导致的 position 漂移问题
```

**面试深挖问题**：
1. 传统 RoPE 在变长序列中有什么问题？
2. Right-Anchored 如何保持时间语义一致？
3. 与绝对位置编码相比的优势？

---

## 第四部分：系统设计与工程实践

### 4.1 Candidate Pipeline 框架

**考察点 16：管道模式设计**

```rust
#[async_trait]
pub trait CandidatePipeline<Q, C> {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    fn selector(&self) -> &dyn Selector<Q, C>;
    
    async fn execute(&self, query: Q) -> PipelineResult<Q, C>;
}
```

**面试深挖问题**：
1. 为什么用 trait 抽象而不是泛型？
2. 如何实现阶段间的并行执行？
3. 如何处理部分失败（如某个 Hydrator 超时）？

### 4.2 异步执行模型

**考察点 17：并行 vs 顺序执行**

```rust
// Hydrators 并行执行
let hydrated = join_all(hydrators.iter().map(|h| h.hydrate(&query, &candidates))).await;

// Filters 顺序执行
let mut result = candidates;
for filter in filters {
    result = filter.filter(&query, result).kept;
}

// SideEffects 异步隔离
tokio::spawn(async move { side_effect.run(input).await; });
```

**面试深挖问题**：
1. 为什么 Hydrators 并行而 Filters 顺序？
2. SideEffects 用 tokio::spawn 的好处？
3. 如何实现背压控制？

### 4.3 错误处理策略

**考察点 18：优雅降级**

```rust
// Hydrator 返回 Vec<Result<C, String>>
// 框架只取 Ok 部分更新
let results: Vec<Result<C, String>> = hydrator.hydrate(&query, &candidates).await;
let ok_candidates: Vec<C> = results.into_iter().filter_map(|r| r.ok()).collect();
```

**面试深挖问题**：
1. 为什么不让整个请求失败？
2. 如何监控和告警降级情况？
3. 如何设计重试策略？

### 4.4 两级管道架构

**考察点 19：组合模式**

```
ForYouCandidatePipeline (第二级)
    └── ScoredPostsSource
            └── PhoenixCandidatePipeline (第一级)
                    ├── ThunderSource
                    ├── PhoenixSource
                    ├── Hydrators (10+)
                    ├── Filters (14+)
                    ├── Scorers (3)
                    └── SideEffects (6+)
```

**面试深挖问题**：
1. 为什么要分两级而不是一级？
2. 如何处理两级之间的数据传递？
3. 如何实现级联超时控制？

---

## 第五部分：面试深挖点与考察细节

### 5.1 模型架构深挖

| 考察点 | 深挖问题 | 期望答案要点 |
|--------|---------|-------------|
| Attention 机制 | tanh clipping 的作用？ | 防止 logit 极端值导致 softmax 梯度消失/爆炸 |
| FFN 设计 | SwiGLU vs ReLU 的区别？ | 门控机制增加表达力，GELU 替代 SiLU 是变体 |
| 位置编码 | RoPE 的优势？ | 相对位置编码，支持外推，计算高效 |
| 模型规模 | 为什么用小模型？ | 推荐系统延迟要求高，小模型更实用 |

### 5.2 训练策略深挖

| 考察点 | 深挖问题 | 期望答案要点 |
|--------|---------|-------------|
| 损失函数 | 多任务学习如何设计损失？ | 加权和，正负反馈分别处理 |
| 负采样 | 如何处理负样本？ | not_interested/block/mute/report 作为硬负样本 |
| 在线学习 | 如何实现实时更新？ | 持续训练，增量更新 embedding |
| 冷启动 | 新用户/新内容如何处理？ | 默认 embedding，快速适配 |

### 5.3 系统设计深挖

| 考察点 | 深挖问题 | 期望答案要点 |
|--------|---------|-------------|
| 延迟优化 | 如何保证 P99 延迟？ | 异步执行，并行 hydrate，缓存 |
| 可扩展性 | 如何添加新特征/模型？ | trait 抽象，插件化设计 |
| 一致性 | 如何保证排序结果一致？ | Candidate Isolation，确定性计算 |
| 可观测性 | 如何监控系统健康？ | tracing，metrics，日志 |

### 5.4 LLM 应用深挖

| 考察点 | 深挖问题 | 期望答案要点 |
|--------|---------|-------------|
| Prompt 设计 | 如何设计有效的 prompt？ | 结构化输出，few-shot，chain-of-thought |
| 输出解析 | 如何处理 LLM 输出不一致？ | Pydantic 验证，fallback，重试 |
| 成本控制 | 如何降低 LLM 调用成本？ | 缓存，批处理，小模型，采样 |
| 延迟优化 | 如何降低 LLM 推理延迟？ | 异步调用，并行请求，模型优化 |

---

## 第六部分：候选人项目介绍模板

### 6.1 一分钟版本

> 我参与了 X (Twitter) 的 For You Feed 推荐系统开发，这是一个基于 Grok Transformer 的端到端推荐系统。系统采用两阶段架构：首先用 Two-Tower 模型从数百万帖子中检索出候选集，然后用 Transformer 模型对候选进行排序。
>
> 我主要负责 LLM 内容理解模块（GROX），利用大语言模型对帖子进行垃圾检测、质量评分和安全分类。这个模块采用异步 DAG 执行架构，支持多级分类和多模态嵌入生成。
>
> 技术亮点包括：基于 LLM 的零样本分类、结构化输出解析、多模态嵌入对齐，以及高并发下的速率限制和优先级调度。

### 6.2 三分钟版本

> 我参与的项目是 X 的 For You Feed 推荐系统，这是一个处理数亿用户、数百万帖子的实时推荐系统。
>
> **系统架构**：采用四级管道设计。第一级是 Thunder，一个基于 Rust 的内存帖子存储，通过 Kafka 实时摄入帖子，支持毫秒级检索。第二级是 Phoenix Retrieval，用 Two-Tower 模型从全局语料库中检索相关帖子。第三级是 Phoenix Ranking，用 Grok-based Transformer 对候选进行排序，预测 19 种用户交互概率。第四级是 Home Mixer，一个 Rust 编写的编排层，负责组装整个推荐管道。
>
> **我的工作**：主要负责 GROX 内容理解模块。这个模块利用 LLM/VLM 对帖子进行深度理解，包括：
> 1. 垃圾评论检测：用 VLM 分析帖子内容和元数据，识别垃圾评论
> 2. 质量评分：给帖子打分，筛选优质内容进入推荐池
> 3. 安全审查：检测违规内容，执行平台政策
> 4. 多模态嵌入：生成帖子的多模态嵌入向量，用于检索和排序
>
> **技术挑战**：
> - 异步 DAG 执行：9 个 Plan 并行执行，每个 Plan 内部有依赖关系
> - 速率限制：用优先级队列和加权随机调度保证公平性
> - 结构化输出：用 Pydantic 解析 LLM 输出，设计 fallback 机制
> - 多模态对齐：处理文本、图片、视频的嵌入对齐问题
>
> **成果**：系统上线后，垃圾评论检测准确率提升 15%，优质内容曝光率提升 20%，用户参与度提升 10%。

### 6.3 技术细节版本（根据面试官追问展开）

**Q: 能详细说说 Candidate Isolation 是怎么实现的吗？**

> Candidate Isolation 是 Phoenix 排序模型的核心设计。在标准 Transformer 中，所有 token 都可以相互 attend。但在推荐排序场景中，我们希望每个候选的得分只依赖于用户历史和自身内容，不依赖于其他候选。
>
> 实现方式是自定义 attention mask。具体来说：
> 1. 首先构建标准的因果 mask，确保历史 token 只能看到之前的 token
> 2. 然后清零 candidate 块的非对角线，阻止候选之间的相互 attend
> 3. 最后恢复对角线，保留候选的自注意力
>
> 这样生成的 mask 结构是：用户可以 attend 所有，历史可以 attend 用户和历史，候选只能 attend 用户、历史和自己。
>
> 这种设计的好处是：
> - 得分独立性：每个候选的得分不依赖于 batch 中有哪些其他候选
> - 可缓存性：得分可以独立缓存，不会因为候选集合变化而失效
> - 一致性：相同用户+相同候选 = 相同得分，无论排序如何

**Q: GROX 的 LLM 分类器是怎么设计的？**

> GROX 的 LLM 分类器采用模板方法模式，分为三个步骤：
>
> 1. `_to_convo`：构建对话。将帖子转换为 LLM 可理解的格式，包括文本、图片、视频帧等多模态信息。对于视频，我们会抽取关键帧并用 ASR 生成转录。
>
> 2. `_sample`：LLM 采样。调用 VLM (Vision-Language Model) 生成分类结果。我们用 temperature=0.000001 来保证输出的确定性。
>
> 3. `_parse`：解析输出。用 Pydantic 模型验证 LLM 输出的 JSON 格式，提取分类结果和置信度。
>
> 以垃圾检测为例：
> - 输入：帖子文本 + 用户元数据（粉丝数、发帖频率等）
> - Prompt：系统提示 + few-shot examples + 帖子内容
> - 输出：`{"is_spam": true, "reason": "重复内容+低粉丝数"}`
> - 后处理：将结果写入 Manhattan 存储，供下游使用
>
> 设计亮点：
> - 两阶段分类：先粗粒度检测是否有违规，再细粒度判定具体违规类型
> - 多模型支持：不同任务用不同大小的模型，平衡成本和效果
> - 失败重试：LLM 调用失败时自动重试，最多 2 次

---

## 第七部分：LLM 通用面试题补充

### 7.1 Transformer 架构

**Q: 解释 Multi-Head Attention 的计算过程**

```
1. 线性投影：Q = XW_Q, K = XW_K, V = XW_V
2. 分头：Q, K, V → [B, T, H, d]
3. 注意力计算：attn = softmax(QK^T / √d) V
4. 拼接：多头输出拼接
5. 输出投影：O = concat(heads) W_O
```

**Q: GQA (Grouped Query Attention) 是什么？为什么用它？**

> GQA 是 MHA 和 MQA 的折中方案。MHA 每个 head 有独立的 K、V，MQA 所有 head 共享 K、V。GQA 将 head 分组，组内共享 K、V。
>
> 优势：
> - 减少 KV cache 大小，降低推理显存
> - 比 MQA 效果更好，因为保留了组内的多样性
> - 训练时可以 MHA → GQA 转换，渐进式减少 KV heads

### 7.2 训练策略

**Q: 解释 RLHF 的流程**

```
1. SFT (Supervised Fine-Tuning)：用高质量数据微调基座模型
2. Reward Model 训练：用人类偏好数据训练奖励模型
3. PPO 优化：用强化学习优化模型，最大化奖励模型分数
```

**Q: DPO (Direct Preference Optimization) 是什么？与 RLHF 的区别？**

> DPO 直接用偏好数据优化策略模型，无需训练奖励模型。
>
> 核心思想：将 RLHF 的约束优化问题转化为简单的分类问题。
>
> 损失函数：L = -log σ(β log π(y_w|x)/π_ref(y_w|x) - β log π(y_l|x)/π_ref(y_l|x))
>
> 优势：
> - 无需训练奖励模型，简化流程
> - 无需在线采样，训练更稳定
> - 超参数更少，更容易调优

### 7.3 推理优化

**Q: KV Cache 是什么？为什么能加速推理？**

> KV Cache 缓存已计算的 Key 和 Value，避免重复计算。
>
> 在自回归生成中，每个新 token 都需要 attend 到所有之前的 token。如果不缓存，每生成一个 token 都要重新计算所有位置的 K、V，复杂度是 O(n²)。缓存后，只需计算新 token 的 Q 和之前所有 K、V 的注意力，复杂度降为 O(n)。

**Q: 解释 PagedAttention**

> PagedAttention 将 KV Cache 分页管理，类似操作系统的虚拟内存。
>
> 核心思想：
> - 将 KV Cache 分成固定大小的 block
> - 用 block table 映射逻辑位置到物理位置
> - 支持非连续存储，减少内存碎片
>
> 优势：
> - 提高显存利用率（从 60% → 90%+）
> - 支持更长的上下文
> - 支持 beam search、parallel sampling 等复杂场景

### 7.4 Agent 与工具使用

**Q: 解释 ReAct 框架**

> ReAct (Reasoning + Acting) 让 LLM 交替进行推理和行动。
>
> 循环过程：
> 1. Thought：推理当前状态，决定下一步行动
> 2. Action：选择工具和参数
> 3. Observation：获取工具返回结果
> 4. 重复直到任务完成
>
> 优势：
> - 可解释性：推理过程透明
> - 灵活性：可以调用外部工具
> - 鲁棒性：可以处理错误和异常

**Q: Function Calling 的实现原理**

```
1. 定义工具 schema（JSON Schema 格式）
2. 将工具定义注入 system prompt
3. LLM 生成结构化的工具调用（JSON 格式）
4. 解析 LLM 输出，执行工具调用
5. 将工具结果反馈给 LLM
```

### 7.5 RAG (Retrieval-Augmented Generation)

**Q: RAG 的基本流程**

```
1. 索引阶段：
   - 文档分块
   - 生成嵌入
   - 存入向量数据库

2. 检索阶段：
   - 用户查询生成嵌入
   - 向量相似度检索
   - 返回 Top-K 文档

3. 生成阶段：
   - 将检索结果注入 prompt
   - LLM 基于上下文生成回答
```

**Q: 如何优化 RAG 的检索质量？**

1. **分块策略**：按语义分块，而非固定长度
2. **嵌入模型**：选择领域适配的嵌入模型
3. **重排序**：用 Cross-Encoder 对检索结果重排序
4. **查询改写**：用 LLM 改写用户查询，提高检索召回
5. **混合检索**：结合向量检索和关键词检索

---

## 附录：关键代码路径参考

### Phoenix 模型
- Transformer 实现：`phoenix/grok.py`
- 排序模型：`phoenix/recsys_model.py`
- 检索模型：`phoenix/recsys_retrieval_model.py`
- 推理 Pipeline：`phoenix/run_pipeline.py`

### Home Mixer
- 主入口：`home-mixer/main.rs`
- 管道定义：`home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs`
- Scorer 实现：`home-mixer/scorers/`
- Filter 实现：`home-mixer/filters/`

### GROX 内容理解
- 主入口：`grox/main.py`
- 分类器：`grox/classifiers/content/`
- 嵌入器：`grox/embedder/`
- 计划系统：`grox/plans/`

### Candidate Pipeline 框架
- 核心 trait：`candidate-pipeline/candidate_pipeline.rs`
- 各阶段 trait：`candidate-pipeline/{source,hydrator,filter,scorer,selector}.rs`

---

## 面试准备 Checklist

### 基础知识
- [ ] Transformer 架构（Attention, FFN, LayerNorm, Position Encoding）
- [ ] LLM 训练流程（Pretrain, SFT, RLHF/DPO）
- [ ] 推理优化（KV Cache, PagedAttention, Quantization）
- [ ] RAG 基本原理

### 项目相关
- [ ] 能画出系统架构图
- [ ] 能解释 Candidate Isolation 的设计动机和实现
- [ ] 能解释 Two-Tower 检索模型的设计
- [ ] 能解释 GROX 的 LLM 分类器设计
- [ ] 能解释 Hash-Based Embedding 的优势

### 系统设计
- [ ] 能解释管道模式的设计
- [ ] 能解释异步执行模型
- [ ] 能解释错误处理策略
- [ ] 能解释延迟优化方法

### 开放问题
- [ ] 如何评估推荐系统的效果？
- [ ] 如何处理冷启动问题？
- [ ] 如何平衡探索和利用？
- [ ] 如何保证推荐的公平性？

---

*最后更新：2026-06-27*
*基于 X (Twitter) For You Feed 推荐系统项目*
