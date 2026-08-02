# anydataset

[English](README.md)

`anydataset` 是一个面向 PyTorch 的数据集抽象层，用来把不同来源的数据统一成同一种逻辑样本结构。数据集产出 `Sample` 映射；`Schema` 描述训练需要哪些 role、模态、视图和字段；`collate_fn(schema)` 把样本整理成可以交给 PyTorch `DataLoader` 使用的 `Batch`。

主流程是：

```text
Spec/Preset -> AnyDataset/IterableAnyDataset -> Sample -> Schema -> collate_fn -> Batch
```

## 安装

```bash
pip install anydataset
```

如果要处理 Hugging Face 数据集或音频文件：

```bash
pip install 'anydataset[huggingface,audio]'
```

本地开发环境：

```bash
pip install -e '.[huggingface,audio,dev]'
```

模型 provider 按需安装。使用 `LongCatProvider` 前安装
`anytrain[longcat]`；使用 `WhisperASRProvider` 或默认语音质量 evaluator 前安装
`anytrain[speech]`；使用 `TextAcceptability` 或 `ChineseGEC` 前安装
`anydataset[text]`；使用
`MossTTSProvider` 前安装 `anytrain[moss-tts]` 和 `anydataset[audio]`。

## 快速开始

```python
from torch.utils.data import DataLoader

from anydataset import AnyDataset
from anydataset.dataset.collate import collate_fn
from anydataset.types import (
    ImageMeta,
    ImageReq,
    ImageView,
    Modality,
    Role,
)

dataset = AnyDataset.preset("mnist", split="train")

schema = {
    (Role.DEFAULT, Modality.IMAGE): ImageReq(
        views=frozenset({ImageView.PIXEL}),
        meta=frozenset({ImageMeta.LABEL}),
    )
}

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn(schema))
batch = next(iter(loader))

image = batch.sample[(Role.DEFAULT, Modality.IMAGE)]
pixels = image.views[ImageView.PIXEL]
labels = image.meta[ImageMeta.LABEL]
```

`pixels` 和 `labels` 都已经是 batch 后的值。普通 Tensor 形状一致时直接 stack；只有最后
一个维度不同时会 pad，并把有效位置记录在 `batch.masks`。waveform 会先通过
`torch.as_tensor` 转成 Tensor，再按同一规则处理。codec view 的单样本 shape 必须是
`[frame, codebook]`，collate 沿 frame 轴 pad，得到 `[batch, frame, codebook]` 和
`[batch, frame]` mask。mapping view 会按 key 递归 batch，并要求 key 和序列长度契约一致；
其他非 Tensor 值返回 list。通用的 dtype 或设备归一化仍应在 dataset transform / preset
parse 阶段完成。

## Cost-aware 动态 batch

map-style `AnyDataset` 可以用 `dataset.dataloader(...)` 做动态 batch。
`costs` 接受 `None`、与 dataset 等长且按全局样本 index 对齐的稳定整数 iterable，
或把 `parse_fn` 同层轻量 row 映射为整数 cost 的 callable。标量整数不接受，
因为常量样本 cost 不携带额外信息；unit-cost batch 请用 `None`。loader 在执行完整 `parse_fn` 前按需读取这些 cost；单条样本 cost
必须是正整数，batch 的 memory 和分布式 compute 都直接使用所选样本 cost 之和。
每个 planning window 内，planner 会贪心选择仍不超过 `max_batch_memory`、且能让当前
batch 尽量填满的样本；`max_batch_samples` 可以额外限制单个 batch 的样本数。

planner 只保留有界 lookahead，提前结束 epoch 时不会读取尚未看见的尾部 cost；完整遍历
epoch 仍必然读取每条所选样本的 cost，因此昂贵长度应在预处理阶段计算并持久化，不能靠
materialize 完整样本临时推导。没有自定义 sampler 时，dataset 会在唯一的 `shuffle`
开关背后生成 rank-local 读取计划，规划后的 batch 不会再被重分配给其他 rank。对
`StoreDataset`，这个私有计划会先 shuffle payload shard group，再把每个 group 内的样本
切分到各 rank 后分别 shuffle，因此即使 store 只有一个 payload shard，也不会让其他 rank
空转；planner 仍只在同一个 shard group 内组 batch。DDP 通过 tensor collective 按有界
plan window 同步，只裁掉 rank-local 的最终 batch 尾部，保证所有 rank step 数一致。
分布式训练每个 epoch 前调用 `loader.set_epoch(epoch)` 推进 shuffle。

## 加载任意数据集

如果数据集已经有内置 preset，优先用 preset：

```python
from anydataset import AnyDataset, IterableAnyDataset

mnist = AnyDataset.preset("mnist", split="train")
fleurs = AnyDataset.preset("fleurs", split="train", config_name="en_us")
```

内置 preset 全部是 map-style：`MNIST`、`CIFAR10`、`FSD50K`、`COMMON_VOICE`、
`FLEURS`、`LIBRISPEECH_ASR`、`ESC50`、`NSYNTH` 和 `WMT19`。
`IterableAnyDataset.preset()` 会显式报错；自定义真正只能流式读取的 source 仍可直接
构造 `IterableAnyDataset`。两个入口都接受
`transforms=...`，在 parse 后执行 item 级 transform。

只需要得到物理 `Spec` 时，直接调用 preset：

```python
from anydataset import Preset

spec = Preset.MNIST.spec(split="train")
```

`FSD50K` 是 map-style preset，只接受可选的 Hugging Face `revision`。revision
默认是 `main`，同时用于文件列表、payload 下载和 source cache identity。

```python
fsd50k = AnyDataset.preset(
    "fsd50k",
    split="dev",
    revision="refs/convert/parquet",
)
```

需要显式指定来源时，使用 `Spec`：

```python
from functools import partial

from anydataset import AnyDataset, Source, Spec
from anydataset.rowmap import sample_from_row
from anydataset.types import ImageMeta, ImageView

dataset = AnyDataset(
    spec=Spec(
        source=Source.HF,
        path="ylecun/mnist",
        split="train",
    ),
    parse_fn=partial(
        sample_from_row,
        image={
            "image": ImageView.PIXEL,
            "label": ImageMeta.LABEL,
        },
    ),
)
```

流式读取的数据集使用 `IterableAnyDataset`：

```python
from functools import partial

from anydataset import IterableAnyDataset, Source, Spec
from anydataset.rowmap import sample_from_row
from anydataset.types import AudioView

dataset = IterableAnyDataset(
    spec=Spec(
        source=Source.HF,
        path="google/fleurs",
        split="train",
        load_options={
            "config_name": "en_us",
        },
    ),
    parse_fn=partial(
        sample_from_row,
        audio={"audio": AudioView.WAVEFORM},
    ),
)
```

当前支持的 source：

- `Source.HF`：通过 `datasets.load_dataset(...)` 读取。
- `Source.HF_DISK`：通过 `datasets.load_from_disk(...)` 读取。
- `Source.HF_FILES`：读取 Hugging Face Hub 上的原始文件树，按需下载文件，
  产出物理文件 row；具体数据集如何映射成 canonical `Sample` 留给 preset 或
  `parse_fn`。
- `Source.STORE`：读取 `anydataset` 的 store。
- 字符串 source `"hf-files"`：对应 `Source.HF_FILES`，可用 `path_prefix`
  或按 split 展开的 `path_template` 限定目录、用 `suffixes` 过滤文件，并返回
  `repo_id`、`repo_type`、`revision`、`path`、`local_path` 等物理字段。
- 字符串 source `"tsv"`：读取单个 TSV 文件、目录下的 `<split>.tsv`，或按
  `subdirs` load option 的顺序读取各子目录下的同名 split。TSV 与 `sharded_csv`
  共用 delimited→Parquet prepare，提供 map-style 随机访问；`root_field` 在读取时注入。
- 字符串 source `"sharded_csv"`：读取 `shard_<index>/<number>.csv` 数字文件名，
  设置 split 时读取 `<path>/<split>/shard_<index>/<number>.csv`；非数字 CSV 文件名
  会被忽略并写 warning。

`anydataset` 的缓存统一放在 `ANYDATASET_HOME` 下；未设置时默认使用
`~/.cache/anydataset`。数据源准备缓存写入
`$ANYDATASET_HOME/cache/sources/<spec_id>`，过滤结果写入
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`。
运行时 warning 和 worker 日志写入
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`。

内置 `tsv` 与 `sharded_csv` source 保留文本表格作为可读的事实来源，并在
`$ANYDATASET_HOME/cache/sources` 下为每个源文件 prepare 一个 Parquet cache part。
prepare 使用 spawn process pool 并行转换变化文件，最后原子提交 cache manifest；
dataset 随后通过 Parquet row group 提供 map-style 随机访问。动态 batch shuffle 会先
打乱 row group 顺序，并且每次只物化一个 row group 的 index，不再分配全数据集 Python
index list。受限环境可以在 `Spec(load_options={"prepare_workers": 0})` 或 `1` 中显式关闭
process pool；默认使用有界的自动并行策略。

只需要得到 `Spec` 时，也可以使用字符串 shorthand：

```python
from anydataset import resolve_dataset

mnist_spec = resolve_dataset("mnist:train")
spec = resolve_dataset("hf://ylecun/mnist:train")
disk_spec = resolve_dataset("hf-disk:///data/mnist_saved:train")
files_spec = resolve_dataset("hf-files://org/files:train")
store_spec = resolve_dataset("store:///data/my_anydataset:train")
tsv_spec = resolve_dataset("tsv:///data/common_voice/en:train")
csv_spec = resolve_dataset("sharded_csv:///data/bitext:train")
```

`Spec.source`、`path`、`split`、`version` 和物理 `load_options` 都参与
`Spec.id`。不改变源内容的 operational 选项（当前为 `prepare_workers`）不进入
`Spec.id` / prepare cache 身份。同一路径改指向另一个物理快照时，应更新 `version`
或对应物理 load option；source prepare cache 只会在最终 identity 相同时复用。

新增物理 source 类型时，注册一个工厂即可；`AnyDataset` 会按 `Spec.source` 从注册器取 source：

```python
from pathlib import Path
from anydataset import IterableAnyDataset, Spec, register_source

class DatabaseSource:
    def prepare(self, spec: Spec, cache_path: Path):
        return open_database_rows(spec.path, **spec.load_options)

register_source("database", DatabaseSource)

dataset = IterableAnyDataset(
    Spec(source="database", path="postgresql://host/db", split="train"),
    parse_fn=parse,
)
```

能够跳过全流扫描的 iterable source 可以额外实现
`iter_shard(dataset, *, num_shards, shard_id)`。该 source 方法必须为精确的
全局 modulo shard 产出 `(sample_index, row)`：索引从 `shard_id` 开始，每次增加
`num_shards`。anydataset 会在 filter 或 materializer 使用前校验 tuple 结构和索引步进；
完整覆盖以及 row 与 index 的对应关系仍由 source 负责。只有 raw dataset 的 `shard()`
或 `iter_shard()` 不足以启用该路径，因为原生 shard 内的局部枚举不能保留全局
索引。

内建 `hf-disk`、`hf-files`、`store`、`tsv` 和 `sharded_csv` source 通过随机访问提供 indexed 路径。
Hugging Face `streaming=True` 会被拒绝；请使用非 streaming 的 `Source.HF`、
`Source.HF_DISK` 或 `Source.HF_FILES`。dataset 层的 `iter_shard` 保留 index，
产出 `(sample_index, sample)`，不会机会主义地调用 raw dataset 的 `shard()`。

## 用 Schema 构造 DataLoader

`Schema` 是从 `(Role, Modality)` 到 requirement 的映射。requirement 指定这个 batch 需要哪些 view 和字段。

这几个概念的分工是：

- `Role` 表达一个 item 在样本里的语义位置，例如 `DEFAULT`、`SOURCE`、`TARGET`。
- `Modality` 表达数据类型，例如 `AUDIO`、`TEXT`、`IMAGE`。
- `View` 表达同一份数据的具体表示，例如音频的 waveform、file、LongCat codes。
- `Meta` 表达标签、语言等旁信息。

分布式训练或多 worker 读取时，可以在 dataset 层做 shard：

```python
rank_iter = dataset.iter_shard(num_shards=8, shard_id=0)
# 产出 (sample_index, sample)
```

```python
from anydataset.types import (
    AudioReq,
    AudioView,
    Modality,
    Role,
)

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
```

然后把 schema 交给 collator：

```python
from torch.utils.data import DataLoader

from anydataset.dataset.collate import collate_fn

loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    collate_fn=collate_fn(schema),
)
```

常用训练输入也建议保留为显式 schema。项目里可以写一个小 helper，然后交给通用
collator：

```python
from anydataset.dataset.collate import collate_fn
from anydataset.types import AudioReq, AudioView, Modality, Role

audio_codec_schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn(audio_codec_schema))
```

一个样本里有多个同模态 item 时，用 role 区分。例如机器翻译可以有 source text 和 target text：

```python
from anydataset.types import (
    Modality,
    Role,
    TextReq,
    TextView,
)

text = TextReq(views=frozenset({TextView.TEXT}))
schema = {
    (Role.SOURCE, Modality.TEXT): text,
    (Role.TARGET, Modality.TEXT): text,
}
```

语音到语音翻译也可以用同一套结构表达。preset 可以产出 source audio 和 target audio；训练时如果只需要 LongCat codes，用户自己写 schema 即可，不需要把这个组合任务放进核心 API：

```python
from anydataset.types import (
    AudioReq,
    AudioView,
    Modality,
    Role,
)

longcat_audio = AudioReq(views=frozenset({AudioView.LONGCAT}))
schema = {
    (Role.SOURCE, Modality.AUDIO): longcat_audio,
    (Role.TARGET, Modality.AUDIO): longcat_audio,
}
```

如果数据集同时提供源语言转写和目标语言文本，可以在 preset 里一起产出文本 item。需要辅助 loss、过滤或调试时，再把文本加进 schema：

```python
from anydataset.types import TextMeta, TextReq, TextView

text = TextReq(
    views=frozenset({TextView.TEXT}),
    meta=frozenset({TextMeta.LANG}),
)
schema = {
    (Role.SOURCE, Modality.AUDIO): longcat_audio,
    (Role.TARGET, Modality.AUDIO): longcat_audio,
    (Role.SOURCE, Modality.TEXT): text,
    (Role.TARGET, Modality.TEXT): text,
}
```

一般来说，preset 负责尽量保留数据集天然提供的信息，schema 负责声明本次训练真正需要的字段。组合型或研究型任务应由用户显式写 schema，不放进核心 API 猜测。

## 缓存过滤分区

`FilterRule` 可以把 map-style `AnyDataset` 按规则分成多个 label，并把每个 label 对应的原始样本下标缓存在 `$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>` 下。predicate 会看到 dataset 产出的完整 canonical `Sample`。

```python
from anydataset.filter import FilterDecision, FilteredDataset, FilterRule

def quality_factory():
    return lambda sample: "review" if needs_review(sample) else is_good(sample)


def dataset_factory():
    return build_dataset()


filtered = FilteredDataset(
    "quality_v1_parse_v3_transform_none",
    quality_factory,
    dataset_factory=dataset_factory,
    device="cpu",
)
train = filtered.select_by("accept")
audit = filtered.select_by("reject", "review")

rule = FilterRule("quality_v1_parse_v3_transform_none", quality_factory)
again = rule.apply(dataset_factory=dataset_factory, labels="accept", device="cpu")
```

predicate 返回 `True` 会归为 `"accept"`，返回 `False` 会归为 `"reject"`；也可以直接返回字符串或枚举值。`name` 是可读规则名，同时为了兼容旧缓存仍参与 identity；可选的 `rule_id` 和 `version` 会增加显式的 predicate、parse function 和 transforms 语义 identity。修改三者中的任一项都会选择不同缓存。省略新字段时 `rule_id` 默认使用 `name`，兼容旧的 name-only 缓存路径。

`FilteredDataset(...)` 会先检查当前 base dataset 和 rule identity 是否已经有可用缓存；没有就先构建。它默认选择缓存里所有 label；需要某些 label 时用 `select_by(...)` 基于同一份缓存派生视图。`FilterRule.apply(...)` 是便利入口，只是把自己的 rule identity 和 `factory` 转发给 `FilteredDataset`。

调用方如果需要观察某一次 apply 的墙钟耗时，可以使用
`FilterRule.apply_with_report(...)`；它不会把本次运行状态存回 dataset 对象：

```python
applied = rule.apply_with_report(dataset_factory=dataset_factory, device="cpu")
filtered = applied.dataset
report = applied.report

if not report.cache_hit and report.logs_dir is not None:
    print(f"filter logs: {report.logs_dir}")
print(
    "filter apply took "
    f"{report.elapsed_seconds:.2f}s "
    f"({report.samples_per_second:.1f} samples/s)"
)
```

`FilterApplyReport` 会拆分 dataset 构建、cache lookup/build 和 partition read
耗时。命中 hot cache 时，`report.logs_dir` 是 `None`，
`report.cache_build_seconds` 是 `0.0`；report 记录的是本次 apply 调用开销，
不是缓存 schema metadata。

物理 dataset 和 filtered view 的 filter cache identity 会自动生成。对于内容或顺序
由业务工程管理、可能变化的输入，调用 `apply()` 或 `FilteredDataset(...)` 时传入非空
`input_id`。它表示整个 filter 输入快照的语义版本，并补充自动生成的 class、`Spec`
和 sample count identity；输入变化时调用方必须更新它。store manifest 的 provenance
会自动参与 identity，因此 materializer 的 `input_id` 或 `provider_id` 变化也会得到新的
filter cache。显式 `input_id` 会由 filtered `dataset_factory`、pickle 和链式过滤继续
携带。

`FilterRule` 保存的是零参数 factory，factory 会在实际执行 predicate 的进程里创建
predicate。`device="auto"` 会解析为全部可见 CUDA device；CUDA 不可用时解析为 CPU。
只解析出一个 device 时在调用进程执行；多个 device 才会按每个 device 一个 worker
启动外层进程，start method 来自 `Runtime.process_start_method`，默认是 `"spawn"`。
传 `device="cpu"` 可以明确使用 CPU；传 `("cpu", "cpu")` 或
`("cuda:0", "cuda:1")` 这样的 iterable 可以显式指定多个 worker。多设备过滤会在
调用 factory 前设置
DDP 常用的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和
`MASTER_PORT` 环境变量，并用 exhaustive 的 runtime 风格 index shard 覆盖每条
base sample。多设备过滤会自己管理这些环境变量，应作为离线预处理运行，不要放进
已经存在的 DDP 训练进程里。数据集入口统一使用 `dataset_factory=...`。使用
`"spawn"` 时，dataset factory 和 predicate factory 都应该是模块顶层的可 pickle
callable。
传 `num_workers` 可以让当前执行进程内部用 PyTorch `DataLoader` 读取样本；
`batch_size` 控制这个 loader 的 batch 大小。

`commit_samples` 控制扫描多少条样本后提交一次内存里的 label batch，默认
100,000；`max_shard_samples` 控制每个 parquet shard 最多多少个下标，默认
1,000,000。这样不会单样本写入，也不会先把几百万个下标全塞进一个 Python 对象或
单个 parquet 文件。filter cache 构建会把已完成 chunk 写进隐藏 resume fragment，
所有样本覆盖后再 replay 成最终 cache。`write_workers` 默认用一个后台 writer，让
predicate 执行和 parquet 写入重叠；`write_prefetch` 控制待写任务上限。

如果 predicate 需要顺手记录逐样本指标，可以返回 `FilterDecision`，并在
`apply` 时显式打开 `metrics=True`：

```python
def metric_factory():
    return lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    )


rule = FilterRule("quality_v2", metric_factory)

filtered = rule.apply(dataset_factory=dataset_factory, metrics=True, device="cpu")
rows = list(filtered.iter_metrics())
```

metrics 会写在 filter cache 下面，每行包含原始样本下标、归一化后的 label
和 JSON 指标 payload。如果旧的分区缓存没有 metrics side output，再次以
`metrics=True` 应用规则时会重建缓存。
完成的 cache 以不可变 generation 发布，并由 reader lease 保护。清理契约和
`cleanup_filter_generations(...)` 入口见
[`docs/filter_cache.md`](docs/filter_cache.md)。

对已经选好的 accept 分区，偶发的廉价 CPU 硬失败可以用
`RejectReplaceDataset` 在线替换（顺序 look-ahead，再回退到 worker 本地 accept
buffer）。它不是缓存分区的替代品；见
[`docs/online_filter.md`](docs/online_filter.md)。

## 质量过滤规则

质量模块提供可复用的 `FilterRule` 规则类；它们不负责加载数据集，也不替调用方
决定缓存 `rule.name`。

文本翻译质量过滤在 `anydataset.quality.translation` 中。内置第一版 profile 面向
WMT19 `zh-en`，输出 `accept`、`review`、`reject` 三类 label。
`FilterRule` 只接受 map-style 输入；WMT19 已是 map-style，可直接过滤：

```python
from anydataset import (
    AnyDataset,
    FilterRule,
    Lang,
    Preset,
)
from anydataset.quality.rules import QualityChain, Rule
from anydataset.quality.text import TextQuality
from anydataset.quality.translation import TranslationQuality
from anydataset.types import Role

source = AnyDataset.preset(
    "wmt19", source_lang="zh", target_lang="en"
)


def dataset_factory():
    return AnyDataset.preset("wmt19", source_lang="zh", target_lang="en")


def translation_factory():
    return QualityChain(
        (
            Rule(
                "source_text",
                TextQuality(role=Role.SOURCE, lang=Lang.ZH),
            ),
            Rule(
                "target_text",
                TextQuality(role=Role.TARGET, lang=Lang.EN),
            ),
            Rule(
                "pair",
                TranslationQuality.from_preset(
                    Preset.WMT19,
                    source_lang=Lang.ZH,
                    target_lang=Lang.EN,
                ),
            ),
        )
    )

filtered = FilterRule("mt_quality_rules_v1_zh_en", translation_factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
train = filtered.select_by("accept")
```

`TextQuality` 只检查单个文本 item，`TranslationQuality` 只检查 source/target
pair 一致性。canonical 语言 meta 使用 `anydataset.Lang`；外部数据集标签在
preset、parser 或 CLI 入口用 `anydataset.remap_lang(...)` 显式映射。

语音质量过滤在 `anydataset.quality.speech` 中。`SpeechQuality` 会检查 canonical
`Sample` 里的每个 audio item，并寻找同 role 的文本作为参考；默认根据 UTMOS、
chrF、秒/文本单位、峰值振幅以及可选 WER/BLEU 阈值输出 `accept` 或 `reject`：

```python
from anydataset import AnyDataset, FilterRule, Source, Spec
from anydataset.quality.speech import SpeechQuality

def speech_dataset_factory():
    return AnyDataset(
        Spec(source=Source.STORE, path="/data/speech-quality-input", split="train")
    )


def speech_factory():
    return SpeechQuality()

filtered = FilterRule("speech_quality_v1", speech_factory).apply(
    dataset_factory=speech_dataset_factory,
    metrics=True,
)
accepted = filtered.select_by("accept")
```

缺少 waveform、同 role 文本等情况会写进 metrics 的 warnings。非有限 waveform 会在
evaluator 执行前直接 reject；其他已检查音频在命中阈值条件时 reject。

## 从 Batch 里取数据

`Batch.sample` 和单条 `Sample` 的逻辑结构相同，只是每个字段都已经 batch 化。

```python
from anydataset.dataset import FieldGroup, FieldRef
from anydataset.types import AudioView, Modality, Role

audio_ref = (Role.DEFAULT, Modality.AUDIO)
audio = batch.sample[audio_ref]

waveform, sample_rate = audio.views[AudioView.WAVEFORM]

waveform_mask = batch.masks[
    FieldRef(
        ref=audio_ref,
        group=FieldGroup.VIEWS,
        key=AudioView.WAVEFORM,
    )
]
```

meta 字段需要先在 schema 里声明，然后从 `item.meta` 里取：

```python
from anydataset.types import ImageMeta

labels = batch.sample[(Role.DEFAULT, Modality.IMAGE)].meta[ImageMeta.LABEL]
```

schema 里声明的 meta 字段必须在 batch 的每条样本中都存在；如果某个数据集不支持该字段，应在 dataset 组合层按任务拆开，而不是让 collator 在同一个 batch 里补空位。非 tensor 值会返回 list。

## Store 和多视图

store 会把样本元信息和 view payload 保存在同一个数据集目录下。同一个模态可以有多个 view。例如音频可以同时有 waveform view、file view、LongCat token view 和 DAC token view。

用 `DatasetWriter` 写出样本：

```python
import torch

from anydataset.store import DatasetWriter
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
)

samples = [
    {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={
                AudioView.WAVEFORM: (torch.tensor([[0.0, 0.1, 0.2]]), 16000),
            },
        )
    }
]

DatasetWriter(
    output_dir="/data/my_anydataset",
    dataset_id="my-audio",
    split="train",
).write(samples)
```

`dataset_id` 可以省略；默认使用展开后的 `output_dir` 最后一段，如果路径没有名称则
使用 `"dataset"`。`provenance` 用于在 dataset manifest 中记录输入的语义状态，只接受
非空字符串 `input_id` 和 `provider_id`。当输入内容或 provider 行为无法由数据路径和配置
完整表达时，用这两个字段显式版本化；它们也会参与下游 filter cache identity。

`write(samples)` 和 `write(dataset_factory=...)` 互斥。默认配置
（`num_shards=1`、`num_workers=0`）在调用进程中串行写入，两种输入形式都可使用。
只要 `num_shards > 1` 或 `num_workers > 0`，就必须传 `dataset_factory`，让每个进程
自行构造 dataset：

```python
from anydataset import AnyDataset
from anydataset.store import DatasetWriter


def dataset_factory():
    return AnyDataset.from_store("/data/source_anydataset")


if __name__ == "__main__":
    DatasetWriter(
        "/data/parallel_copy",
        provenance={"input_id": "source-2026-07-30"},
        num_shards=4,
        num_workers=2,
        prefetch_factor=4,
    ).write(dataset_factory=dataset_factory)
```

`num_shards` 是 store 写进程数；`num_workers` 是每个写进程内部的 PyTorch
`DataLoader` 读取进程数，因此完整并行时除写进程外最多还会启动
`num_shards * num_workers` 个 loader workers。`prefetch_factor` 表示每个 loader
worker 预取的一条样本 batch 数，只在 `num_workers > 0` 时生效，省略时默认为 `2`。
并行 factory 必须可 pickle；默认 `spawn` 启动方式下通常应定义为模块顶层 callable，
返回的 dataset 必须是 map-style，或实现 `iter_shard()`。应从应用主进程调用
writer。当启用 loader workers 且 dataset 暴露 `prepare()` 时，writer 会在 spawn 前
先在父进程调用一次，以便安全准备共享的 source 元数据。

只选择训练需要的 view 读回来：

```python
from anydataset import AnyDataset
from anydataset.types import AudioView, Modality, Role

dataset = AnyDataset.from_store(
    "/data/my_anydataset",
    split="train",
    views=((Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),),
)
```

`AnyDataset.from_store(..., views=...)` 只加载所选 view 的 manifest 和 payload；选择会在
pickle/spawn 后保留，并参与 filter cache identity，而物理 `Spec` 仍只标识 store 本身。

store payload 按 view 写入 tar shard。普通 `DataLoader(shuffle=True)` 会在样本级打散，
一个 batch 可能频繁跨 tar 读取。训练 store 时使用同一个
`dataset.dataloader(..., shuffle=True)` 入口即可保持 local-aware：

```python
from anydataset.dataset.collate import collate_fn

loader = dataset.dataloader(
    costs=lengths,
    max_batch_memory=64_000,
    planning_window=256,
    distributed_plan_window=32,
    max_batch_samples=32,
    shuffle=True,
    seed=13,
    collate_fn=collate_fn(schema),
)
```

当底层 reader 是 `StoreDataset` 时，loader 会按已选择 view 的 payload shard group
生成读取计划：先 shuffle shard group 顺序，再把每个 group 内的样本切分到各 rank 后分别
shuffle；`StoreDataset.__getitem__` 和全局 index shard 语义不变。外层只保留
`dataset.dataloader(..., shuffle=...)` 这一处
shuffle 配置，不再需要单独导入 store 专用 sampler。
DDP 下 `distributed_plan_window` 控制每次同步前各 rank 需要先生成多少个本地 batch plan；
当 cost lookup 或 packing 较重时，可以调小它来降低 first-batch 等待。
设置 `ANYDATASET_DEBUG_DDP_PLANS=1` 可以在每个 DDP planning chunk 同步前后输出
rank、窗口和 plan count。

reader 显式支持 `schema_version: 2` 和当前的 `schema_version: 3`。v2 store 没有
provenance，仍可直接读取；新写入的 v3 store 会保存 materializer 的 `input_id` 和
`provider_id`。

> 警告：v2 兼容只是旧数据读取支持，不是推荐的生产格式。读取 v2 store 会发出
> `RuntimeWarning`；缺失的 provenance 会按空值参与下游 cache identity，无法区分
> input/provider 的语义版本。发布 store 或基于它生成 cache-sensitive 派生数据前，
> 请重新物化或迁移到 v3。

更早的 canonical store 使用相同的 sample manifest 和目录布局，但 dataset manifest
没有版本号，view manifest 使用 `sample_id` 对齐，必须离线迁移到新目录：

```bash
anydataset-store migrate /data/my_anydataset_v1 /data/my_anydataset_v3
```

等价 Python 入口是
`anydataset.store.migrate_store("/data/my_anydataset_v1", "/data/my_anydataset_v3")`。

更早的目录布局，或不完整匹配该 canonical schema 的 v1 manifest，不做猜测式迁移，
必须用 `DatasetWriter` 从原始 canonical dataset 重新物化。

发布或迁移 store 前可以使用公开 integrity API。`fast` 校验 manifest 引用和 shard
存在性，`normal` 还会解析所有被引用的 tar 并拒绝非法或重复的 regular file member，
默认的 `full` 进一步核对每个 manifest payload key：

```python
from pathlib import Path
from anydataset.store import validate_store_payloads

validate_store_payloads((Path("/data/my_anydataset"),), level="full")
```

reader 会保留 manifest 和 tar 句柄以复用随机读取；生命周期明确时使用
`dataset.close()`，或把 `read_store_dataset(...)` 作为 context manager 使用。

`AudioView.FILE` payload 会解包到 `$ANYDATASET_HOME/cache/store-files`。选择了 file
view 的 reader 在自身生命周期内自动持有共享 lease，因此只要 reader 仍可达，cleanup
就不能让已经返回的路径失效。如果路径需要比 reader 活得更久，显式持有 lease；确认没有
reader 或显式 lease 后，再按物理 store 清理：

```python
from anydataset.store import cleanup_store_files, lease_store_files

with lease_store_files("/data/my_anydataset"):
    retained_path = dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
    del dataset
    consume(retained_path)

cleanup_store_files("/data/my_anydataset")
```

等价命令是 `anydataset-store cleanup-files /data/my_anydataset`。即使活动 reader 位于
其他进程，cleanup 也会显式报错而不是删除 leased 文件。该缓存不做自动淘汰；显式清理后，
下一次访问会重新解包。

训练时需要哪些 view，仍然由 schema 指定：

```python
from anydataset.types import (
    AudioReq,
    AudioView,
    Modality,
    Role,
)

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
```

## 生成新的 View

store 的 view 目录直接使用 `{role}/{modality}/{view}`，真实 payload 放在该 view 目录下的
`shards/` 里。`ViewMaterializer` 总是发布可以独立读取的 store。默认只写 provider 的
输出 view；输入 view 和 meta 只有在 `keep_schema` 中显式声明时才会保留。

```python
import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.store import ViewMaterializer
from anydataset.types import AudioView

class ToyLongCat:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, sample_rate = views[AudioView.WAVEFORM]
        return waveform.transpose(0, 1).to(torch.int64)

def dataset_factory():
    return AnyDataset(
        Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
    )


def provider_factory(device: str):
    return ToyLongCat()


output = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
    input_id="my-audio-v1",
    provider_id="toy-longcat-v1",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="cpu",
)

dataset = AnyDataset(
    Spec(source=Source.STORE, path=str(output), split="train"),
)
```

如果输出需要携带少量输入字段，用现有 schema 契约显式声明，不复制整个 sample：

```python
from anydataset.types import Modality, Role, TextMeta, TextReq, TextView

keep_schema = {
    (Role.DEFAULT, Modality.TEXT): TextReq(
        views=frozenset({TextView.TEXT}),
        meta=frozenset({TextMeta.LANG}),
    )
}
materializer = ViewMaterializer(
    "/data/my_anydataset_longcat",
    keep_schema=keep_schema,
)
```

`keep_schema` 声明的字段必须存在；如果其中的 view 与 provider 输出冲突，会显式报错而
不是覆盖。需要多个派生 view 时，对上一步的 standalone store 再运行 materializer，并
显式保留下一步需要的字段。

如果 provider 需要 GPU，可以用 `devices` 控制执行设备。`devices="auto"` 会解析为
全部可见 CUDA device；CUDA 不可用时解析为 CPU。只解析出一个 device 时在调用进程
执行；多个 device 才会按每个 device 一个 worker 启动外层进程，start method 来自
`Runtime.process_start_method`，默认是 `"spawn"`。多设备 worker 分别写 fragment 和
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/materializer/part-xxxxx.log`，全部完成后主进程
合并 store。materializer 默认使用可续跑 fragment：已完成的
provider batch 会聚合成 checkpoint chunk，保留在目标目录旁边的隐藏 resume 目录中；
重跑时按全局 `sample_index` 跳过，最后再原子提交最终 store。`commit_samples`
控制 checkpoint 粒度，默认是 `max(batch_size, 1024)`，避免默认可续跑时产生过多小文件；
需要更细断点时可以显式调低。
resume compatibility 会记录两个 factory 的自动标识。当输入快照或 provider 行为依赖
callable 无法表达的状态（例如可变文件或 checkpoint 内容）时，用 `input_id` 和
`provider_id` 显式给出语义版本。这两个 ID 会补充而不是替代 factory 标识；任一 ID
变化都会隔离旧 resume 目录，不会复用不兼容的 fragment。它们也会写入最终 store
manifest 的 provenance，参与下游 filter cache identity。
和过滤一样，多设备 materialize 拥有自己的离线 worker，不应放进已经存在的 DDP
训练进程里运行。
如果 `parse_fn` 里有 file 到 waveform 这类 CPU 重活，可以给 materializer 传
`num_workers`，让每个设备 worker 内部通过 PyTorch `DataLoader` 做读取、
解码和预取。materializer 会为设备 worker 设置 rank 环境，dataset 的 runtime
shard 会把 rank 和 DataLoader worker 组合起来，保证样本只覆盖一次。
`write_workers` 控制每个 materializer worker 内部的后台写线程数，默认用一个 writer
让 provider 计算和 fragment 落盘重叠；`write_prefetch` 控制待写任务上限。
公开 API 默认值刻意保持保守：默认路径要在不同 provider、平台和调用环境里都能以
单进程、单样本方式稳定运行。生产 workflow 应在脚本或 job wrapper 中根据 provider、
存储后端和硬件显式调 `batch_size`、`num_workers`、`prefetch_factor`、
`write_workers`、`write_prefetch` 和 `commit_samples`。

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatProvider

    return LongCatProvider(device=device)


output = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
    batch_size=8,
    num_workers=4,
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)
```

多设备 materialize 使用配置的 process start method；默认 `"spawn"` 下 factory 应放在
模块顶层，不能用 lambda 或局部函数。

`write()` 必须从应用主进程调用，不能从 PyTorch `DataLoader` worker 或其他 daemon
进程调用。多设备模式会为每张卡创建一个显式 non-daemon 的 materializer 进程；每个
materializer 进程可以再创建 `num_workers` 个 DataLoader reader。切换 `fork` 或
`spawn` 不能解除 Python 对 daemon 进程创建子进程的限制。

需要让 provider 以 batch 调模型时，给 materializer 传 `batch_size`，并在
provider 上实现 `call_batch(batch)`。`batch` 是 `collate_fn` 返回的
`Batch(sample, masks)`。`batch_size=1` 走单条 `__call__`；`batch_size>1` 强制要求
`call_batch`，缺少时会显式抛出 `TypeError`。`Batch.masks` 是通用有效位置表达，序列长度可以用
`batch.lengths(field_ref)` 从 mask 派生。view 或 modality materializer 只 batch
单个输入引用时，`call_batch` 可以直接返回一组输出；同一个 batch 里有多个输入
引用时，`call_batch` 必须返回从 `(role, modality)` 引用到该引用输出序列的映射。

如果 batch provider 抛出 out-of-memory，materializer 会清理可用缓存，并递归把当前 batch
拆成两个更小的 batch 重试。每次拆分都会立即写入包含 worker、provider 类型、失败 batch
大小和重试大小的进度日志。单条样本本身触发 OOM 时无法恢复，会直接向调用方抛出异常。

LongCat provider 的 batch 路径会把 waveform 或 file 输入 padding 后交给 LongCat
encoder。同一个 batch 里有多个 audio role 时，它会在同一个 collated batch 里按
role 分别 encode。file batch 会先在 audio provider 层加载成 waveform；因为 file
view 没有 mask，有效长度来自加载后的 waveform。当前 LongCat encoder 不接收 mask，
所以 provider 会根据每个输入 waveform 的有效长度按比例裁剪输出 codes，避免把
padding 对应的 codes 写入 store。

`write()` 可以并行写 part store 后统一 commit。`num_shards` 控制写进程数，
`num_workers` 控制每个写进程内部的 `DataLoader` workers；并行写入时建议传入
模块顶层的 `dataset_factory`，避免 spawn worker pickle 已构造 dataset 实例。

`ViewMaterializer` 和 `ModalityMaterializer` 的最终输出都可以直接作为
`Source.STORE` 输入，不需要再和 base store 组合。

生成的新 view 也通过 schema 选择：

```python
schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.LONGCAT}),
    )
}
```

## 生成新的 Modality

`ModalityMaterializer` 在同一个 role 下补齐缺失 modality。provider 声明输出 view，
materializer 由它推导输出 modality，并要求该 role 恰好只剩一个输入 modality。已有输出
modality 或输入歧义都会显式报错；生成 item 的 meta 默认为空。

```python
from anydataset.store import ModalityMaterializer
from anydataset.types import AudioView, TextView


class ToyTTS:
    output = AudioView.WAVEFORM

    def __call__(self, views):
        return synthesize(views[TextView.TEXT])


def tts_provider_factory(_device: str):
    return ToyTTS()


output = ModalityMaterializer(
    output_dir="/data/my_anydataset_tts",
).write(
    dataset_factory=dataset_factory,
    provider_factory=tts_provider_factory,
    devices="cpu",
)
```

`MossTTSProvider` 用于 text-to-audio，`WhisperASRProvider` 用于 audio-to-text。
provider 还可以设置 `reference_role`，表示生成时需要该 role 已经存在的输出 modality，
例如 TTS 的参考音频。reference role 不再作为输出目标，其 view 会与其他 role 唯一的输入
modality 一起交给 provider。

## LongCat Provider

LongCat 可以作为可选 provider 使用。provider 会加载
`anytrain.codec.longcat.LongCat`。输出 view 是整数 Tensor：单样本 shape 为
`[frame, codebook]`，collate 后为 `[batch, frame, codebook]`，对应 mask 为
`[batch, frame]`。`CodecProvider` 生成 view 时按照 codec 契约逐列检查输出，第 k 列
的每个 id 必须满足 `0 <= id < codebook_sizes[k]`。store manifest 不保存
`codebook_sizes`，所以 reader 和 collate 不对直接读取的 store view 重复检查值域。
公共数据层不区分 semantic / acoustic codebook；需要解释具体码本语义时，由下游任务按
codec 契约处理。waveform 输入的采样率来自
`AudioView.WAVEFORM` 的 `(waveform, sample_rate)` value，file 输入的采样率来自
`torchaudio.load()`。

典型流程是先用 preset 或 source 读出 waveform store，再 materialize 成可以独立读取的
LongCat store，并在训练 schema 里选择 `AudioView.LONGCAT`：

```text
base waveform store -> ViewMaterializer + LongCatProvider -> standalone store -> schema selects LONGCAT
```

```python
from anydataset import AnyDataset, Source, Spec
from anydataset.provider.longcat import LongCatProvider

output = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)

dataset = AnyDataset(
    Spec(source=Source.STORE, path=str(output), split="train"),
)
```

## 开发

```bash
python -m compileall -q src tests examples
python -m ruff check src tests scripts examples
python -m basedpyright --pythonpath "$(python -c 'import sys; print(sys.executable)')"
python -m pytest -q
```

设计说明在 [docs/design.md](docs/design.md)，filter cache 细节在
[docs/filter_cache.md](docs/filter_cache.md)，在线 reject 替换说明在
[docs/online_filter.md](docs/online_filter.md)，质量过滤说明在
[docs/translation_quality.md](docs/translation_quality.md) 和
[docs/speech_quality.md](docs/speech_quality.md)，远端模型进程和 runtime 配置在
[docs/provider_service.md](docs/provider_service.md)，性能验证结果在
[docs/experiments/results/](docs/experiments/results/)。

## 发布

```bash
python scripts/check_release.py
```

`anydataset` v1 把 canonical `Sample` 映射、source registry、filter cache
布局、store schema 和 materializer API 作为公开稳定面。包会暴露
`anydataset.__version__`，发布检查会先确认它和 `pyproject.toml` 版本一致，
再清理旧构建产物、运行 pytest、构建 sdist/wheel、执行 `twine check`，并在
隔离虚拟环境里安装 wheel 做 smoke test。只想检查版本和测试门禁时，可以加
`--skip-build`。
