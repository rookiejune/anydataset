# anydataset

[English](README.md)

`anydataset` 是一个面向 PyTorch 的数据集抽象层，用来把不同来源的数据统一成同一种逻辑样本结构。数据集产出 `Sample` 映射；`Schema` 描述训练需要哪些 role、模态、视图和字段；`collate_fn(schema)` 把样本整理成可以交给 PyTorch `DataLoader` 使用的 `Batch`。

主流程是：

```text
Spec/Preset -> AnyDataset/IterableAnyDataset -> Sample -> Schema -> collate_fn -> Batch
```

## 安装

```bash
pip install -e '.[huggingface,test]'
```

如果要处理音频数据集：

```bash
pip install -e '.[huggingface,audio]'
```

## 快速开始

```python
from torch.utils.data import DataLoader

from anydataset import (
    ImageMeta,
    ImageReq,
    ImageView,
    Modality,
    Preset,
    Role,
    collate_fn,
)

dataset = Preset.MNIST.create(split="train")

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

`pixels` 和 `labels` 都已经是 batch 后的值。collator 只会 batch 已经是 `torch.Tensor` 的字段；形状一致时会直接 stack，如果只有最后一个维度长度不同，会 pad 到最长长度，并把有效位置记录在 `batch.masks` 中。dict view 会按 key 分别 batch，默认单条 sample 内各 key 的 tensor 最后一个维度长度一致。需要转 tensor、统一 dtype 或设备时，请在 dataset transform / preset parse 阶段完成。

## 加载任意数据集

如果数据集已经有内置 preset，优先用 preset：

```python
from anydataset import Preset

mnist = Preset.MNIST.create(split="train")
fleurs = Preset.FLEURS.create(split="train", config_name="en_us")
```

当前内置 preset 包括 `MNIST`、`CIFAR10`、`FLEURS`、`LIBRISPEECH_ASR`、
`COMMON_VOICE`、`ESC50`、`NSYNTH`、`FSD50K` 和 `WMT19`。

需要显式指定来源时，使用 `Spec`：

```python
from functools import partial

from anydataset import (
    AnyDataset,
    ImageMeta,
    ImageView,
    Source,
    Spec,
)
from anydataset.utils import sample_from_row

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

from anydataset import AudioView, IterableAnyDataset, Source, Spec
from anydataset.utils import sample_from_row

dataset = IterableAnyDataset(
    spec=Spec(
        source=Source.HF,
        path="google/fleurs",
        split="train",
        load_options={
            "config_name": "en_us",
            "streaming": True,
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
- `Source.STORE`：读取 `anydataset` 的 store。
- 字符串 source `"tsv"`：读取单个 TSV 文件、目录下的 `<split>.tsv`，或按
  `subdirs` load option 的顺序读取各子目录下的同名 split。
- 字符串 source `"sharded_csv"`：读取 `shard_<index>/<number>.csv` 数字文件名，
  设置 split 时读取 `<path>/<split>/shard_<index>/<number>.csv`；非数字 CSV 文件名
  会被忽略并写 warning。

`anydataset` 的缓存统一放在 `ANYDATASET_HOME` 下；未设置时默认使用
`~/.cache/anydataset`。数据源准备缓存写入
`$ANYDATASET_HOME/cache/sources/<spec_id>`，过滤结果写入
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`。
运行时 warning 和 worker 日志写入
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`。

只需要得到 `Spec` 时，也可以使用字符串 shorthand：

```python
from anydataset import resolve_dataset

spec = resolve_dataset("hf://ylecun/mnist:train")
disk_spec = resolve_dataset("hf-disk:///data/mnist_saved:train")
store_spec = resolve_dataset("store:///data/my_anydataset:train")
tsv_spec = resolve_dataset("tsv:///data/common_voice/en:train")
csv_spec = resolve_dataset("sharded_csv:///data/bitext:train")
```

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

## 组合数据集

`MultipleAnyDataset` 可以把多个数据集组合成一个 iterable dataset。组合后的迭代顺序由 strategy 决定。

```python
from anydataset import MultipleAnyDataset, Preset, RoundRobinStrategy

dataset = MultipleAnyDataset(
    datasets=[
        Preset.FLEURS.create(split="train", config_name="en_us"),
        Preset.LIBRISPEECH_ASR.create(split="train.100"),
    ],
    strategy=RoundRobinStrategy(),
)
```

按权重随机采样：

```python
from anydataset import MultipleAnyDataset, Preset, WeightedRandomStrategy

dataset = MultipleAnyDataset(
    datasets=[
        Preset.FLEURS.create(split="train", config_name="en_us"),
        Preset.LIBRISPEECH_ASR.create(split="train.100"),
    ],
    strategy=WeightedRandomStrategy(weights=[1.0, 2.0], seed=13),
)
```

分布式训练或多 worker 读取时，可以在 dataset 层做 shard：

```python
rank_iter = dataset.shard(num_shards=8, shard_id=0)
```

## 用 Schema 构造 DataLoader

`Schema` 是从 `(Role, Modality)` 到 requirement 的映射。requirement 指定这个 batch 需要哪些 view 和字段。

这几个概念的分工是：

- `Role` 表达一个 item 在样本里的语义位置，例如 `DEFAULT`、`SOURCE`、`TARGET`。
- `Modality` 表达数据类型，例如 `AUDIO`、`TEXT`、`IMAGE`。
- `View` 表达同一份数据的具体表示，例如音频的 waveform、file、LongCat codes。
- `Meta` 表达标签、语言等旁信息。

```python
from anydataset import AudioReq, AudioView, Modality, Role

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
```

然后把 schema 交给 collator：

```python
from torch.utils.data import DataLoader

from anydataset import collate_fn

loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    collate_fn=collate_fn(schema),
)
```

如果内置 task 的默认 schema 已经够用，也可以直接用 `Task`：

```python
from anydataset import Task

schema = Task.AUDIO_CODEC.schema()
loader = DataLoader(dataset, batch_size=16, collate_fn=Task.AUDIO_CODEC.collate_fn())
```

一个样本里有多个同模态 item 时，用 role 区分。例如机器翻译可以有 source text 和 target text：

```python
from anydataset import Modality, Role, TextReq, TextView

text = TextReq(views=frozenset({TextView.TEXT}))
schema = {
    (Role.SOURCE, Modality.TEXT): text,
    (Role.TARGET, Modality.TEXT): text,
}
```

语音到语音翻译也可以用同一套结构表达。preset 可以产出 source audio 和 target audio；训练时如果只需要 LongCat codes，用户自己写 schema 即可，不需要为这个组合任务新增内置 `Task`：

```python
from anydataset import AudioReq, AudioView, Modality, Role

longcat_audio = AudioReq(views=frozenset({AudioView.LONGCAT}))
schema = {
    (Role.SOURCE, Modality.AUDIO): longcat_audio,
    (Role.TARGET, Modality.AUDIO): longcat_audio,
}
```

如果数据集同时提供源语言转写和目标语言文本，可以在 preset 里一起产出文本 item。需要辅助 loss、过滤或调试时，再把文本加进 schema：

```python
from anydataset import TextMeta, TextReq, TextView

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

一般来说，preset 负责尽量保留数据集天然提供的信息，schema 负责声明本次训练真正需要的字段。内置 `Task` 只适合非常稳定、跨数据集一致的默认 schema；组合型或研究型任务建议由用户显式写 schema。

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

predicate 返回 `True` 会归为 `"accept"`，返回 `False` 会归为 `"reject"`；也可以直接返回字符串或枚举值。`FilterRule` 的缓存契约就是用户提供的 `name`。predicate、parse function 和 transforms 的语义版本由调用方写进 `name`。

`FilteredDataset(...)` 会先检查当前 base dataset 和 rule name 是否已经有可用缓存；没有就先构建。它默认选择缓存里所有 label；需要某些 label 时用 `select_by(...)` 基于同一份缓存派生视图。`FilterRule.apply(...)` 是便利入口，只是把自己的 `name` 和 `factory` 转发给 `FilteredDataset`。

`FilterRule` 保存的是零参数 factory，factory 会在实际执行 predicate 的进程里创建
predicate。`device="auto"` 会在有可见 CUDA 时每张卡启动一个 spawn 进程，没有
CUDA 时退回 CPU 单进程。传 `device="cpu"` 可以明确使用 CPU 单进程；传
`("cpu", "cpu")` 或 `("cuda:0", "cuda:1")` 这样的 iterable 可以显式指定多个
worker。多设备过滤会为每个 device 启动一个固定 worker，在调用 factory 前设置
DDP 常用的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和
`MASTER_PORT` 环境变量，并用 exhaustive 的 runtime 风格 index shard 覆盖每条
base sample。多设备过滤会自己管理这些环境变量，应作为离线预处理运行，不要放进
已经存在的 DDP 训练进程里。数据集入口统一使用 `dataset_factory=...`。
dataset factory 和 predicate factory 都应该是模块顶层的可
pickle callable。
传 `num_workers` 可以让每个设备进程内部用 PyTorch `DataLoader` 读取样本；
`batch_size` 控制这个 loader 的 batch 大小。

`commit_samples` 控制扫描多少条样本后提交一次内存里的 label batch，默认
100,000；`max_shard_samples` 控制每个 parquet shard 最多多少个下标，默认
1,000,000。这样不会单样本写入，也不会先把几百万个下标全塞进一个 Python 对象或
单个 parquet 文件。

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

## 质量过滤 Predicate

质量模块提供可复用的 `FilterRule` predicate；它们不负责加载数据集，也不替调用方
决定缓存 `rule.name`。

文本翻译质量过滤在 `anydataset.quality.translation` 中。内置第一版 profile 面向
WMT19 `zh-en`，输出 `clean`、`usable`、`review`、`reject` 四类 label：

```python
from anydataset import FilterRule, Preset
from anydataset.quality.translation import Predicate as TranslationQuality

dataset = Preset.WMT19.create(source_lang="zh", target_lang="en")
def translation_factory():
    return TranslationQuality.from_preset(
        Preset.WMT19,
        source_lang="zh",
        target_lang="en",
    )

filtered = FilterRule("mt_quality_rules_v1_zh_en", translation_factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
train = filtered.select_by("clean", "usable")
```

语音质量过滤在 `anydataset.quality.speech` 中。predicate 会检查 canonical
`Sample` 里的每个 audio item，并寻找同 role 的文本作为参考；默认根据 UTMOS、
chrF、秒/文本单位、峰值振幅以及可选 WER/BLEU 阈值输出 `accept` 或 `reject`：

```python
from anydataset import FilterRule
from anydataset.quality.speech import Predicate as SpeechQuality

def speech_factory():
    return SpeechQuality()

filtered = FilterRule("speech_quality_v1", speech_factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
accepted = filtered.select_by("accept")
```

缺少 waveform、同 role 文本等情况会写进 metrics 的 warnings；当前规则只在已经
检查到的音频低于阈值时 reject。

## 从 Batch 里取数据

`Batch.sample` 和单条 `Sample` 的逻辑结构相同，只是每个字段都已经 batch 化。

```python
from anydataset import AudioView, FieldGroup, FieldRef, Modality, Role

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
from anydataset import ImageMeta

labels = batch.sample[(Role.DEFAULT, Modality.IMAGE)].meta[ImageMeta.LABEL]
```

schema 里声明的 meta 字段必须在 batch 的每条样本中都存在；如果某个数据集不支持该字段，应在 dataset 组合或 `IterationStrategy` 层按任务拆开，而不是让 collator 在同一个 batch 里补空位。非 tensor 值会返回 list。

## Store 和多视图

store 会把样本元信息和 view payload 保存在同一个数据集目录下。同一个模态可以有多个 view。例如音频可以同时有 waveform view、file view、LongCat token view 和 DAC token view。

用 `DatasetWriter` 写出样本：

```python
import torch

from anydataset import AudioItem, AudioView, DatasetWriter, Modality, Role

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

用 `Source.STORE` 读回来：

```python
from anydataset import AnyDataset, Source, Spec

dataset = AnyDataset(
    spec=Spec(
        source=Source.STORE,
        path="/data/my_anydataset",
        split="train",
    ),
)
```

训练时需要哪些 view，仍然由 schema 指定：

```python
from anydataset import AudioReq, AudioView, Modality, Role

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
```

## 生成新的 View

store 的 view 目录直接使用 `{role}/{modality}/{view}`，真实 payload 放在该 view 目录下的 `shards/` 里。`ViewMaterializer` 会读取已有 dataset，把每个 item 的全部 views 交给 provider，由 provider 决定如何生成自己的输出 view。它写出的是 delta store：保留样本和轻量 meta，只写 provider 的输出 view，不复制原来的 view payload。base store 和 delta store 按 `sample_index` 对齐；调用方负责保证派生 view 来自同一顺序的数据集。

```python
import torch

from anydataset import AnyDataset, AudioView, Source, Spec, ViewMaterializer

class ToyLongCat:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, sample_rate = views[AudioView.WAVEFORM]
        return {
            "semantic_codes": waveform.to(torch.int64),
        }

def dataset_factory():
    return AnyDataset(
        Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
    )


def provider_factory(device: str):
    return ToyLongCat()


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="cpu",
)

merged = AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta), split="train"))
)

merged.write("/data/my_anydataset_with_longcat", split="train")
```

如果 provider 需要 GPU，可以用 `devices` 控制并行设备。`devices="auto"` 会
检测当前可见 CUDA 设备，每张卡启动一个 spawn worker；每个 worker 写自己的
part 和 `$ANYDATASET_HOME/logs/<timestamp>-<pid>/materializer/part-xxxxx.log`，
全部完成后主进程合并 store。
和过滤一样，多设备 materialize 拥有自己的离线 worker，不应放进已经存在的 DDP
训练进程里运行。
如果 `parse_fn` 里有 file 到 waveform 这类 CPU 重活，可以给 materializer 传
`num_workers`，让每个设备 worker 内部通过 PyTorch `DataLoader` 做读取、
解码和预取。materializer 会为设备 worker 设置 rank 环境，dataset 的 runtime
shard 会把 rank 和 DataLoader worker 组合起来，保证样本只覆盖一次。

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatProvider

    return LongCatProvider(device=device)


delta = ViewMaterializer(
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

多设备 materialize 使用 Python `spawn`，所以 factory 应放在模块顶层，不能用
lambda 或局部函数。

需要让 provider 以 batch 调模型时，给 materializer 传 `batch_size`，并在
provider 上实现 `call_batch(batch)`。`batch` 是 `collate_fn` 返回的
`Batch(sample, masks)`；`batch_size=1` 或 provider 没有 batch 方法时会继续走
单条 `__call__` 路径。`Batch.masks` 是通用有效位置表达，序列长度可以用
`batch.lengths(field_ref)` 从 mask 派生。view 或 modality materializer 只 batch
单个输入引用时，`call_batch` 可以直接返回一组输出；同一个 batch 里有多个输入
引用时，`call_batch` 必须返回从 `(role, modality)` 引用到该引用输出序列的映射。

LongCat provider 的 batch 路径会把 waveform 或 file 输入 padding 后交给 LongCat
encoder。同一个 batch 里有多个 audio role 时，它会在同一个 collated batch 里按
role 分别 encode。file batch 会先在 audio provider 层加载成 waveform；因为 file
view 没有 mask，有效长度来自加载后的 waveform。当前 LongCat encoder 不接收 mask，
所以 provider 会根据每个输入 waveform 的有效长度按比例裁剪输出 codes，避免把
padding 对应的 codes 写入 store。

`merge()` 返回逻辑合并后的 map-style dataset，不修改左右两侧的物理 store。它按
相同下标取左右两侧样本，相当于 `zip(strict=True)`；左右两侧必须都是 map-style
dataset，并且长度必须一致。右侧可以提供新 item 或同一 item 下的新 view；重复 view
会报错，重复 metadata 只有值相等时才允许。调用方负责维护两个 dataset 的顺序、
过滤和版本一致性。runtime shard 会作用在合并后的 dataset 上，因此两侧会共享同一个
全局下标。

需要发布自包含 store 时，对逻辑合并结果显式调用 `write()`：

```python
merged.write(
    "/data/my_anydataset_with_longcat",
    split="train",
)
```

`write()` 可以并行写 part store 后统一 commit。`num_shards` 控制写进程数，
`num_workers` 控制每个写进程内部的 `DataLoader` workers；并行写入时建议传入
模块顶层的 `dataset_factory`，避免 spawn worker pickle 已构造 dataset 实例：

```python
merged.write(
    "/data/my_anydataset_with_longcat",
    split="train",
    num_shards=4,
    num_workers=4,
    dataset_factory=merged_dataset_factory,
)
```

生成的新 view 也通过 schema 选择：

```python
schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.LONGCAT}),
    )
}
```

## LongCat Provider

LongCat 可以作为可选 provider 使用。provider 会加载 `anytrain.codec.longcat.LongCatAudioCodec`。输出 view 只保存 codes；waveform 输入的采样率来自 `AudioView.WAVEFORM` 的 `(waveform, sample_rate)` value，file 输入的采样率来自 `torchaudio.load()`。

典型流程是先用 preset 或 source 读出 waveform store，再 materialize 成 LongCat delta store，合并后在训练 schema 里选择 `AudioView.LONGCAT`：

```text
base waveform store -> ViewMaterializer + LongCatProvider -> delta store -> logical merge -> schema selects LONGCAT
```

```python
from anydataset import AnyDataset, Source, Spec
from anydataset.provider.longcat import LongCatProvider

delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)

merged = AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta), split="train"))
)

merged.write("/data/my_anydataset_with_longcat", split="train")
```

## 开发

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

设计说明在 [docs/design.md](docs/design.md)，filter cache 细节在
[docs/filter_cache.md](docs/filter_cache.md)，质量过滤说明在
[docs/translation_quality.md](docs/translation_quality.md) 和
[docs/speech_quality.md](docs/speech_quality.md)，待办事项在 [todo.md](todo.md)。
