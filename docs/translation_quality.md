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
- 第一版实现放在 `examples/translation_quality_filter.py`，稳定后再考虑抽到
  `src/anydataset/quality/translation.py`。

## 标签

过滤规则输出三个 label：

- `accept`：高置信保留，可直接进入训练。
- `review`：结构上可疑但不一定错误，训练时可降权或交给后续模型复核。
- `reject`：高置信丢弃。

不要只保留一个布尔结果。对于 2kw 级别数据，`review` 分区能支持后续抽样、
阈值校准和神经网络补打分。

## 第一版级联

第一版只做轻量规则和可解释 metrics：

1. 空文本、控制字符、过长重复字符。
2. 源文和译文长度比例异常。
3. 源文和译文在不同语言时完全相同。
4. 数字、日期、占位符、HTML 标签的一致性。
5. 数字表面形式是否保真，例如 `6.0` 和 `6` 的数值相同，但在训练数据里
   可能应该保留格式差异，默认进入 `review`。
6. 根据语言提示检查主要文字脚本，例如英文应主要是 Latin，中文应主要是 CJK。

这些规则不会证明一句翻译是好翻译，只负责便宜地发现明显脏数据和需要复核的
灰区。

## Metrics

每条样本的 metrics 至少包含：

- `source_chars`、`target_chars`、`char_ratio`
- `source_lang`、`target_lang`
- `source_script_ratio`、`target_script_ratio`
- `number_value_overlap`、`number_surface_overlap`
- `flags`
- `quality_score`

`FilterRule.apply(..., metrics=True)` 会把这些 metrics 缓存在 filter cache 下。
后续可以从 `FilterResult.iter_metrics()` 抽样查看，也可以按 flags 或 score 做
二次分析。

## 神经网络后端

神经网络不进入第一层默认实现。需要时按成本从低到高增加：

1. 跨语言 embedding 相似度，例如 LaBSE 或 LASER。这个分数只能命名为
   `semantic_score`，表示语义相似，不表示样本适合训练。
2. bitext classifier，例如 Bicleaner AI。
3. reference-free QE，例如 COMETKiwi。
4. LLM 只用于抽检、阈值校准和 disagreement 样本复核。

推荐只对 `review`、模型分歧样本和每个语种/领域/长度桶的抽样数据使用神经网络。
如果要在复旦服务器上跑模型，按仓库根目录 `docs/fdu-remote.md` 的约定使用
`ssh 121`，并把 Hugging Face 缓存显式放到 `/mnt/pami202/zhuyin/huggingface`。

神经网络分数不能覆盖硬规则。以下情况应由规则优先处理：

- 占位符不同，例如 `{price}` 变成 `{amount}`。
- 数值不同，例如 `12` 变成 `13`。
- 数字表面形式改变，例如 `6.0` 变成 `6`。
- 单位、货币、日期、代码片段或 HTML 标签丢失。

其中数值不同默认 `reject`；表面形式不同默认 `review`，对于要求格式严格保真的
训练任务可以提升为 `reject`。embedding 分数即使很高，也不应把这些样本自动改回
`accept`。

## Cache Name

`FilterRule.name` 是缓存契约。任何会改变 label 或 metrics 语义的内容都应该写进
name，例如：

```text
mt_quality_rules_v1_en_zh_len_0p2_6p0_parser_v1
mt_quality_cometkiwi_wmt22_v1_en_zh_rules_v1
```

不要依赖库自动检查 predicate、parse function、模型权重或阈值是否变化。

## Example

本地调试可以从 TSV/CSV 文本对开始：

```bash
python examples/translation_quality_filter.py \
  --input storage/translation_pairs.tsv \
  --source-column source \
  --target-column target \
  --source-lang zh \
  --target-lang en \
  --cache-root storage/translation-quality-cache \
  --rule-name mt_quality_rules_v1_zh_en
```

这个 example 自带一个轻量 map-style 表格 source，只用于本地小样本调试。大规模
数据应优先进入 Hugging Face map-style dataset 或 `anydataset` store，再复用相同
predicate 和 `FilterRule`。
