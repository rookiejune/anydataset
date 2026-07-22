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

过滤规则输出三个面向训练选择的 label：

- `accept`：当前规则和已接入模型均未触发排除；适合进入训练候选集，但不表示规则已经证明语义翻译正确。
- `review`：前面规则给出 `reject`，后续规则给出 `accept`，表示规则之间存在分歧；用于人工抽样、LLM 复核或更强模型复核。
- `reject`：当前 pipeline 排除。它可能来自任一原子规则的明确拒绝，或后续模型规则低分。

实现上不需要扩展 `FilterRule`。质量规则直接返回上述字符串 label，filter
缓存会按 label 建分区。

不同训练目标可以选择不同 label：

- 普通机器翻译训练：使用 `accept`。
- 高召回清洗或阈值校准：可抽样查看 `review`，但不要默认并入训练集。
- 质检抽样：重点看 `review`、模型分歧 flag，以及所有 `reject` 的误杀率。
- 严格格式保真任务：仍然从 `accept` 里按 `flags` 排除表面格式变化样本。

## Decision Order

统一规则按优先级决策，而不是把所有指标揉成一个分数：

1. **输入和结构硬错误**：缺失或非字符串文本、空文本、控制字符、过长重复字符、
   明显文字脚本错误、placeholder 或 HTML tag multiset 不一致。这类默认 `reject`。
2. **简单数值硬错误**：当前 number parser 能明确解析的数字 token 不一致，例如
   `12 -> 13`，默认 `reject`。
3. **表面格式差异**：数值等价但表面形式不同，例如 `6.0 -> 6`，仍输出
   `accept`，并记录 `number_surface_mismatch` flag。
4. **复杂数字表达**：年代、世纪、约数、自然语言数量单位、`early/mid/late`
   等规则难以可靠判断的表达，由 pair 原子规则返回 `reject`。
5. **结构异常**：长度比例异常或轻度文字脚本偏差由对应原子规则返回 `reject`。
6. **未命中规则**：输出 `accept`，但这只表示轻量规则没有发现问题，不是语义证明。
7. **后续规则**：Bicleaner 等模型规则返回 `accept` 时可以把已有 `reject` 提升到
   `review`；返回 `reject` 时可以把当前 `accept` 或 `review` 降为 `reject`。

典型例子：

```text
12 -> 13                         reject
{price} -> {amount}              reject
6.0 -> 6                         accept + number_surface_mismatch flag
1.9万 -> 19,000                  reject，后续模型 accept 时变成 review
19世纪30年代 -> 1830s            reject，后续模型 accept 时变成 review
19世纪30年代 -> late 1830s       reject，后续模型 accept 时变成 review
```

当前实现不解析单位、货币、日期字段或代码语法，也不比较这些 token 的语义。例如
`12公斤 -> 12 pounds`、`12美元 -> 12 euros` 或 `foo() -> bar()` 可能因为数字一致且
其他轻量规则未命中而得到 `accept`。这些约束应由调用方增加专用规则或交给后端复核，
不能从当前 label 反推它们已经一致。

## 第一版级联

第一版只做轻量规则和可解释 audit flags：

1. 空文本、控制字符、过长重复字符。
2. 源文和译文长度比例异常。
3. 源文和译文在不同语言时完全相同。
4. 简单数字 token、占位符和 HTML 标签的一致性；复杂数字表达是可恢复 reject。
5. 数字表面形式是否保真，例如 `6.0` 和 `6` 的数值相同，但在训练数据里
   可能应该保留格式差异，默认 `accept` 并写入 `number_surface_mismatch`。
6. 根据语言提示检查主要文字脚本，例如英文应主要是 Latin，中文应主要是 CJK。

这些规则不会证明一句翻译是好翻译，只负责便宜地发现明显脏数据、可直接候选数据和
需要模型或人工复核的灰区。

## Audit Log

`FilterDecision.metrics` 是给人抽样审计看的短日志，不作为完整指标表。第一版每条
样本只写入：

- `source`、`target`
- `decision`
- `source_lang`、`target_lang`
- `flags`

`source_lang`、`target_lang` 在 canonical sample 和 quality profile 中使用
`anydataset.Lang`。外部数据集标签例如 `en_us`、`zh-CN` 必须在 preset、parser
或 CLI 入口用 `anydataset.remap_lang(...)` 显式映射后再进入 quality 规则。

`FilterRule.apply(..., metrics=True)` 会把这些 audit row 缓存在 filter cache 下。
后续可以从 `FilteredDataset.iter_metrics()` 抽样查看，也可以按 flags 做人工复核。
规则内部用单文本 `quality._text.Metrics` 和文本对 `_Metrics` 分层；具体指标用
`cached_property` 按需计算并在子规则之间共享，但默认不写入日志。
规则层不输出泛化的 `quality_score`。如果接入 Bicleaner 或其他模型，分数应按来源
命名，例如 `bicleaner_score`、`semantic_score` 或 `qe_score`。

## 神经网络后端

神经网络不进入轻量规则默认链。需要时按成本从低到高增加：

1. 单语句子可接受性模型，接入 `TextAcceptability(role, lang)`。英文默认使用
   `textattack/roberta-base-CoLA`；没有默认模型的语种会显式报错，等库内新增
   对应 `Lang -> model` 映射后再启用。使用默认模型前安装 `anydataset[text]`。
2. 中文 GEC 反推质量，接入 `ChineseGEC(role)`。默认使用
   `shibing624/mengzi-t5-base-chinese-correction` 生成纠错后文本，并按
   `gec_edit_count` / `gec_edit_ratio` 判断；它不是 acceptability classifier。
3. 跨语言 embedding 相似度，例如 LaBSE 或 LASER。这个分数只能命名为
   `semantic_score`，表示语义相似，不表示样本适合训练。
4. bitext classifier，例如 Bicleaner AI，分数命名为 `bicleaner_score`。
5. reference-free QE，例如 COMETKiwi。
6. LLM 只用于抽检、阈值校准和 disagreement 样本复核。

推荐只对可恢复 reject、模型分歧样本和每个语种/领域/长度桶的抽样数据使用
神经网络。
如果要在远程 GPU 机器上跑模型，把 Hugging Face 缓存、模型版本、阈值和推理参数
显式写进调用方配置，并把会影响 label 的部分同步写进 `FilterRule.name`。

神经网络分数不能直接把样本升回 `accept`。当前轻量规则优先处理明确配对错误：

- 占位符不同，例如 `{price}` 变成 `{amount}`。
- 可明确解析的简单数值不同，例如 `12` 变成 `13`。
- HTML tag multiset 不同。

单位、货币、日期和代码语义不属于当前规则能力，必须由调用方或其他 evaluator 负责。
简单数值不同默认 `reject`；表面形式不同默认 `accept` 加 flag。链式入口里，
后续规则返回 `accept` 时可以把此前的 `reject` 升成 `review`，但不能直接升回
`accept`；后续规则返回 `reject` 会把当前 `review` 再打回 `reject`。

## Implementation Shape

`anydataset.quality` 暴露语义类名。单文本、文本对和模型规则都是原子规则，
入口用 `QualityChain` 显式指定执行顺序：

```python
from anydataset import FilterRule, Lang, Preset
from anydataset.quality.rules import QualityChain, Rule
from anydataset.quality.text import ChineseGEC, TextAcceptability, TextQuality
from anydataset.quality.translation import Bicleaner, TranslationQuality
from anydataset.types import Role

def factory():
    return QualityChain(
        (
            Rule(
                "source_text",
                TextQuality(role=Role.SOURCE, lang=Lang.ZH),
            ),
            Rule(
                "source_zh_gec",
                ChineseGEC(role=Role.SOURCE, max_edit_ratio=0.05),
            ),
            Rule(
                "target_text",
                TextQuality(role=Role.TARGET, lang=Lang.EN),
            ),
            Rule(
                "target_acceptability",
                TextAcceptability(
                    role=Role.TARGET,
                    lang=Lang.EN,
                    min_score=0.7,
                ),
            ),
            Rule(
                "pair",
                TranslationQuality.from_preset(
                    Preset.WMT19,
                    source_lang=Lang.ZH,
                    target_lang=Lang.EN,
                ),
            ),
            Rule(
                "bicleaner",
                Bicleaner.from_preset(
                    Preset.WMT19,
                    source_lang=Lang.ZH,
                    target_lang=Lang.EN,
                    scorer=bicleaner_score,
                    min_score=0.6,
                ),
            ),
        )
    )

rule = FilterRule("mt_quality_rules_v1_wmt19_zh_en", factory)
```

链式组合器负责 label 转移：`accept -> reject` 变成 `reject`，`reject -> accept`
变成 `review`，`review -> reject` 变成 `reject`。每条规则自己的 label、flags 和
metrics 都会写入最终 audit row 的 `rules` 字段；顶层 `flags` 使用
`rule_name:flag` 形式保留具体来源。Bicleaner 只按 `min_score` 输出
`bicleaner_accept` 或 `bicleaner_reject`。

## Cache Name

`FilterRule.name` 是缓存契约。任何会改变 label 或 audit row 语义的内容都应该写进
name，例如：

```text
mt_quality_rules_v1_en_zh_len_0p2_6p0_parser_v1
mt_quality_accept_review_reject_v1_en_zh_parser_v1
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
