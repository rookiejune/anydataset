# Translation Quality Filter

本文档记录文本翻译句对质量过滤的第一版方案。它的目标是用
`anydataset.filter` 已有的分区和 metrics 缓存能力，先跑通低成本闭环；
神经网络模型只作为可选后端进入灰区样本或高价值数据。

## 边界

- 质量判别作用在 canonical `Sample` 上，默认读取
  `(Role.SOURCE, Modality.TEXT)` 和 `(Role.TARGET, Modality.TEXT)` 的
  `TextView.TEXT`。
- `Source` 仍然只负责物理数据读取，不判断任务或字段语义。
- `Preset` 只负责把原始字段映射为 source/target text，不内置质量规则。
- `anydataset.filter` 只负责执行 predicate、缓存 label 分区和 metrics，不承载
  翻译业务规则或神经网络依赖。
- 第一版规则实现放在 `src/anydataset/quality/translation.py`；
  `examples/translation_quality_filter.py` 只保留本地 TSV/CSV 调试入口。

## Label Contract

过滤规则输出四个面向训练选择的 label：

- `clean`：简单、一目了然、硬约束全一致，适合做最干净的核心训练集。
- `usable`：翻译可用，但有自然改写、轻微格式差异或规则无法证明它是最干净
  样本；普通翻译训练可以使用，严格格式保真任务可降权或排除。
- `review`：可能合理，但规则无法可靠评价；用于人工抽样、LLM 复核或更强模型
  复核。
- `reject`：高置信不适合训练。

不要只保留一个布尔结果，也不要把 `accept` 同时表示“可用”和“很干净”。
对于 2kw 级别数据，`clean` 可作为高质量核心集，`usable` 可作为扩大训练集的
候选，`review` 用于阈值校准和神经网络补打分。

实现上不需要扩展 `FilterRule`。predicate 直接返回上述字符串 label，filter
缓存会按 label 建分区。

不同训练目标可以选择不同 label：

- 严格格式保真任务：只用 `clean`。
- 普通机器翻译训练：使用 `clean` + `usable`。
- 大规模弱监督或预训练：可加入低风险 `review`，但应保留采样权重或来源标记。
- 质检抽样：重点看 `usable` 与 `review` 的边界，以及所有 `reject` 的误杀率。

## Decision Order

统一规则按优先级决策，而不是把所有指标揉成一个分数：

1. **安全和结构硬错误**：空文本、语言明显错误、控制字符、严重乱码、占位符丢失、
   代码片段损坏、HTML 标签明显不一致。这类默认 `reject`。
2. **数值硬错误**：简单可解析数字、百分比、金额或单位数值不一致，例如
   `12 -> 13`，默认 `reject`。
3. **表面格式差异**：数值等价但表面形式不同，例如 `6.0 -> 6`，输出
   `usable`。
4. **简单且约束全一致**：长度比例正常、脚本正常、数字/占位符/HTML 等硬约束一致，
   且没有复杂时间、单位或语义压缩扩展，输出 `clean`。
5. **自然改写但低风险**：语义看起来可用，但有轻微改写、压缩、扩展或格式变化，
   输出 `usable`。
6. **复杂但可能合理**：年代、世纪、约数、自然语言数量单位、`early/mid/late`
   等规则难以可靠判断的表达，输出 `review`。
7. **神经后端回调**：规则层先给出当前 label；Bicleaner 等后端接收当前 label
   后直接返回新 label。高分可以把规则层 `reject` 拉回 `review`，但不会自动升到
   `clean`。

典型例子：

```text
12 -> 13                         reject
{price} -> {amount}              reject
6.0 -> 6                         usable
1.9万 -> 19,000                  review，等 number parser 能证明等值后再升级
19世纪30年代 -> 1830s            review
19世纪30年代 -> late 1830s       review
```

## 第一版级联

第一版只做轻量规则和可解释 audit flags：

1. 空文本、控制字符、过长重复字符。
2. 源文和译文长度比例异常。
3. 源文和译文在不同语言时完全相同。
4. 数字、日期、占位符、HTML 标签的一致性。
5. 数字表面形式是否保真，例如 `6.0` 和 `6` 的数值相同，但在训练数据里
   可能应该保留格式差异，默认进入 `usable`。
6. 根据语言提示检查主要文字脚本，例如英文应主要是 Latin，中文应主要是 CJK。

这些规则不会证明一句翻译是好翻译，只负责便宜地发现明显脏数据、最干净数据和
需要复核的灰区。

## Audit Log

`FilterDecision.metrics` 是给人抽样审计看的短日志，不作为完整指标表。第一版每条
样本只写入：

- `source`、`target`
- `decision`
- `source_lang`、`target_lang`
- `flags`

`FilterRule.apply(..., metrics=True)` 会把这些 audit row 缓存在 filter cache 下。
后续可以从 `FilteredDataset.iter_metrics()` 抽样查看，也可以按 flags 做人工复核。
规则内部仍有一个继承文本对的 `_Metrics` 上下文；具体指标用 `cached_property`
按需计算并在子规则之间共享，但默认不写入日志。
规则层不输出泛化的 `quality_score`。如果接入 Bicleaner 或其他模型，分数应按来源
命名，例如 `bicleaner_score`、`semantic_score` 或 `qe_score`。

## 神经网络后端

神经网络不进入第一层默认实现。需要时按成本从低到高增加：

1. 跨语言 embedding 相似度，例如 LaBSE 或 LASER。这个分数只能命名为
   `semantic_score`，表示语义相似，不表示样本适合训练。
2. bitext classifier，例如 Bicleaner AI，分数命名为 `bicleaner_score`。
3. reference-free QE，例如 COMETKiwi。
4. LLM 只用于抽检、阈值校准和 disagreement 样本复核。

推荐只对 `usable`、`review`、模型分歧样本和每个语种/领域/长度桶的抽样数据使用
神经网络。
如果要在远程 GPU 机器上跑模型，把 Hugging Face 缓存、模型版本、阈值和推理参数
显式写进调用方配置，并把会影响 label 的部分同步写进 `FilterRule.name`。

神经网络分数不能直接把样本升回 `clean`。以下情况应由规则优先处理：

- 占位符不同，例如 `{price}` 变成 `{amount}`。
- 数值不同，例如 `12` 变成 `13`。
- 数字表面形式改变，例如 `6.0` 变成 `6`。
- 单位、货币、日期、代码片段或 HTML 标签丢失。

其中数值不同默认 `reject`；表面形式不同默认 `usable`。Bicleaner 分数很高时，
如果当前 label 是 `reject`，最多拉回 `review`，交给人工或后续抽样检查。

## Implementation Shape

`anydataset.quality.translation` 暴露短名对象，语义由模块路径补充：

```python
from anydataset import FilterRule, Preset
from anydataset.quality.translation import Predicate

def factory():
    return Predicate.from_preset(
        Preset.WMT19,
        source_lang="zh",
        target_lang="en",
        bicleaner=bicleaner_score,
    )

rule = FilterRule("mt_quality_rules_v1_wmt19_zh_en", factory)
```

内部 label 是有序枚举：`reject < review < usable < clean`。每个私有子规则返回
`(ceiling, matched)`；如果命中，就把当前样本评级和 `ceiling` 取最小值，并把
子规则名写入 `flags`。Bicleaner 作为 callback 接收当前 label：

- `bicleaner_score >= 0.7`：高置信互译；当前为 `reject` 时拉回 `review`，否则保持。
- `0.6 <= bicleaner_score < 0.7`：最多保留到 `usable`；当前为 `reject` 时拉回
  `review`。
- `bicleaner_score < 0.6`：输出 `reject`。

callback 会把 `bicleaner_score` 写入 audit row，并追加 `bicleaner_high`、
`bicleaner_usable` 或 `bicleaner_reject` flag。

## Cache Name

`FilterRule.name` 是缓存契约。任何会改变 label 或 audit row 语义的内容都应该写进
name，例如：

```text
mt_quality_rules_v1_en_zh_len_0p2_6p0_parser_v1
mt_quality_clean_usable_review_reject_v1_en_zh_parser_v1
mt_quality_bicleaner_v1_wmt19_zh_en_0p6_0p7_rules_v1
mt_quality_cometkiwi_wmt22_v1_en_zh_rules_v1
```

不要依赖库自动检查 predicate、parse function、模型权重或阈值是否变化。

## Example

本地调试可以从 TSV/CSV 文本对开始：

```bash
ANYDATASET_HOME=storage/anydataset-home \
python examples/translation_quality_filter.py \
  --input storage/translation_pairs.tsv \
  --source-column source \
  --target-column target \
  --source-lang zh \
  --target-lang en \
  --rule-name mt_quality_rules_v1_zh_en
```

这个 example 自带一个轻量 map-style 表格 source，只用于本地小样本调试。大规模
数据应优先进入 Hugging Face map-style dataset 或 `anydataset` store，再复用相同
predicate 和 `FilterRule`。
