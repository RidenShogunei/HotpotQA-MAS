# MAS v2 错误分析报告

## 评测结果

| 指标 | Single Agent (v2) | MAS (v2) | 变化 |
|------|-------------------|----------|------|
| Exact Match | **46.0%** | **42.0%** | ↓ -4.0% |
| Average F1 | **55.2%** | **52.2%** | ↓ -3.0% |
| Avg Latency | 2,430ms | 6,262ms | ↑ +158% |
| Avg Tokens | 248 | 511 | ↑ +106% |

## Failure Mode 分布

| 失败模式 | 数量 | 占比 |
|---------|------|------|
| none (成功) | 21 | 42% |
| main_synthesis_failed | 20 | 40% |
| wrong_answer | 5 | 10% |
| no_evidence | 4 | 8% |

## 核心发现

### main_synthesis_failed (20/50 = 40%) 的根因

**→ 19/20 (95%) 是因为 Sub Agent 检索到了错误的文档!**
**→ 只有 1/20 是因为 Main Agent 自己幻觉了答案**

## 具体失败模式分析

### 1. Sub Agent 检索质量差 (核心问题)

- 搜索关键词提取不准确
- 文档相关性判断错误
- 19/20 的失败都是因为 Sub 检索到错误文档

**典型案例:**

**案例 1: Task 6576 (检索完全错误)**
- Q: What team was formed in 1996 that starred players like Robert Parish?
- Gold: 50 Greatest Players in NBA History
- Sub 1 搜索 "Robert Parish" → 检索到 "Pyramid Valley" (新西兰岩石!)
- → Sub Agent 读了这个完全无关的文档，答案自然错误
- → Main Agent 被误导，输出 "Boston Celtics"

**案例 2: Task 5711 (检索到无关文档)**
- Q: Robin R. Bottin collaborated with director who won Oscar/Golden Globe/BAFTA for what movie?
- Gold: The Social Network
- Sub 1 搜索 "David Fincher" → 检索到 "The Pilot (Friends)"
- Sub 2 搜索 "Rob Bottin" → 检索到 "Danny Boyle"
- → 两个 Sub 都没检索到相关文档
- → Main 输出 "Schindler's List" (幻觉)

### 2. Subtask 设计问题

- Subtask 2 总是 "How does X relate to answering Y?"
- 这种设计导致搜索查询过长，包含整个问题
- 搜索引擎无法处理长查询，返回不相关结果

### 3. Main Agent 过度依赖 Sub

- Main Agent 没有验证 Sub 答案的能力
- 即使 Sub 返回明显错误的答案，Main 也直接采用
- 缺乏交叉验证机制

### 4. 训练数据问题

- SFT 数据可能过于简化，没有教模型如何处理错误信息
- 缺乏 "识别错误检索结果" 的训练样本

## 成功案例 vs 失败案例对比

| 指标 | 成功案例 | 失败案例 |
|------|---------|---------|
| Sub Agent 文档相关率 | 48.7% | 42.4% |

差距不大，说明即使成功案例的检索质量也不高，只是碰巧蒙对了。

## 与 Single Agent 对比

**Single Agent (46% EM):**
- 直接搜索，自己判断文档相关性
- 虽然检索也可能失败，但不会被别人误导
- 错误模式: no_evidence (检索不到) 或 wrong_answer (理解错)

**MAS (42% EM):**
- 分工导致信息传递失真
- Sub 的错误会传递给 Main，且 Main 无法纠正
- 错误模式: main_synthesis_failed (Sub 提供错误信息)

## 结论

**对于 9B 模型，Single Agent 更可靠，因为:**

1. 减少了信息传递环节
2. 避免了 "错误信息被采纳" 的问题
3. 检索失败时可以直接说 "不知道"，而不是编造答案

**MAS 的问题不是 "分工不够细"，而是 "信息传递失真"。** 在小模型上，多增加一个推理环节就多一份出错的可能，且模型没有足够能力验证和纠正错误。

## 改进方向 (如果需要继续优化 MAS)

1. **改进 Sub Agent 的检索策略**
   - 训练更好的关键词提取
   - 增加文档相关性判断能力

2. **改进 Subtask 设计**
   - 避免 "How does X relate to Y" 这种无意义的问题
   - 让 Subtask 更具体、更直接

3. **增加验证机制**
   - Main Agent 验证 Sub Agent 的答案是否合理
   - 不一致时重新委托或自己回答

4. **训练数据改进**
   - 增加 "识别错误检索结果" 的样本
   - 教模型如何处理冲突信息
