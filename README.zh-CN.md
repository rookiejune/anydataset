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

准备数据源时的缓存根目录默认是 `~/.cache/anydataset`。如果希望缓存放到项目自己的 `storage/`、`outputs/` 或其它目录，可以设置 `ANYDATASET_CACHE_ROOT`，也可以在 dataset 构造函数里传 `cache_root`。

只需要得到 `Spec` 时，也可以使用字符串 shorthand：

```python
from anydataset import resolve_dataset

spec = resolve_dataset("hf://ylecun/mnist:train")
disk_spec = resolve_dataset("hf-disk:///data/mnist_saved:train")
store_spec = resolve_dataset("store:///data/my_anydataset:train")
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

`FilterRule` 可以把 map-style `AnyDataset` 按规则分成多个 label，并把每个 label 对应的原始样本下标缓存在物理 `Spec` 的 cache 目录下。predicate 会看到 dataset 产出的完整 canonical `Sample`。

```python
from anydataset.filter import FilterDecision, FilterRule, FilteredDataset

rule = FilterRule(
    name="quality_v1_parse_v3_transform_none",
    predicate=lambda sample: "review" if needs_review(sample) else is_good(sample),
)

train = FilteredDataset(dataset, rule, labels="accept", num_workers=4)
audit = FilteredDataset(dataset, rule, labels=("reject", "review"))
```

predicate 返回 `True` 会归为 `"accept"`，返回 `False` 会归为 `"reject"`；也可以直接返回字符串或枚举值。`FilterRule` 的缓存契约就是用户提供的 `name`。predicate、parse function 和 transforms 的语义版本由调用方写进 `name`。

`FilteredDataset` 会先检查当前 base dataset 和 rule name 是否已经有可用缓存；没有就先构建，再只暴露调用方在 `labels` 里指定的分区。

filter 构建默认单进程；传入 `num_workers` 后会按 map-style 下标范围并行扫描。`commit_samples` 控制扫描多少条样本后提交一次内存里的 label batch，默认 100,000；`max_shard_samples` 控制每个 parquet shard 最多多少个下标，默认 1,000,000。这样不会单样本写入，也不会先把几百万个下标全塞进一个 Python 对象或单个 parquet 文件。

如果 predicate 需要顺手记录逐样本指标，可以返回 `FilterDecision`，并在
`apply` 时显式打开 `metrics=True`：

```python
rule = FilterRule(
    name="quality_v2",
    predicate=lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    ),
)

result = rule.apply(dataset, metrics=True)
rows = list(result.iter_metrics())
```

metrics 会写在 filter cache 下面，每行包含原始样本下标、归一化后的 label
和 JSON 指标 payload。如果旧的分区缓存没有 metrics side output，再次以
`metrics=True` 应用规则时会重建缓存。

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
    cache_root="/data/my_anydataset_cache",
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

store 的 view 目录直接使用 `{role}/{modality}/{view}`，真实 payload 放在该 view 目录下的 `shards/` 里。`ViewMaterializer` 会读取已有 dataset，把每个 item 的全部 views 交给 provider，由 provider 决定如何生成自己的输出 view。它写出的是 delta store：保留样本和轻量 meta，只写 provider 的输出 view，不复制原来的 view payload。

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
        cache_root="/data/my_anydataset_cache",
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

AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta), split="train"))
)
```

如果 provider 需要 GPU，可以用 `devices` 控制并行设备。`devices="auto"` 会
检测当前可见 CUDA 设备，每张卡启动一个 spawn worker；每个 worker 写自己的
part 和 `<output_dir>/logs/part-xxxxx.log`，全部完成后主进程合并 store。

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatViewProvider

    return LongCatViewProvider(device=device)


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)
```

多设备 materialize 使用 Python `spawn`，所以 factory 应放在模块顶层，不能用
lambda 或局部函数。

`merge()` 会把右侧 dataset 里的新 view 或新 item 原地合入左侧 store；如果 view 或 metadata 已经存在且发生冲突，会直接报错。也可以把 delta store 当作目标 store，执行 `AnyDataset(Spec(source=Source.STORE, path=str(delta), split="train")).merge(dataset)`，避免先复制一份 source。

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
base waveform store -> ViewMaterializer + LongCatViewProvider -> delta store -> AnyDataset(store).merge(delta) -> schema selects LONGCAT
```

```python
from anydataset import AnyDataset, Source, Spec
from anydataset.provider.longcat import LongCatViewProvider

delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    split="train",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)

AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta), split="train"))
)
```

## 开发

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

设计说明在 [docs/design.md](docs/design.md)，待办事项在 [todo.md](todo.md)。
