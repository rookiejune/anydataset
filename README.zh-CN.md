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

`pixels` 和 `labels` 都已经是 batch 后的值。collator 只会 batch 已经是 `torch.Tensor` 的字段；形状一致时会直接 stack，如果只有最后一个维度长度不同，会 pad 到最长长度，并把有效位置记录在 `batch.masks` 中。需要转 tensor、统一 dtype 或设备时，请在 dataset transform / preset parse 阶段完成。

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

store 的 view 目录直接使用 `{role}/{modality}/{view}`，真实 payload 放在该 view 目录下的 `shards/` 里。`ViewMaterializer` 会读取已有 dataset，把每个 item 的全部 views 交给 provider，由 provider 决定如何生成自己的输出 view。

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

dataset = AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset", split="train"),
    cache_root="/data/my_anydataset_cache",
)

ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    dataset_id="my-audio",
    split="train",
).write(dataset, ToyLongCat())
```

默认情况下，`ViewMaterializer` 会写出一个 view-only 数据集：它保留样本和轻量 meta，只写 provider 的输出 view，不复制原来的 view payload。不同 provider、参数或实验版本由调用方通过 `output_dir`、provider 输出 view、目录命名和实验文档区分。

如果希望输出目录是完整独立的数据集，同时带有原 view 和新 view，使用 `copy_inputs=True`：

```python
ViewMaterializer(
    output_dir="/data/my_anydataset_with_longcat",
    dataset_id="my-audio",
    split="train",
    copy_inputs=True,
).write(dataset, ToyLongCat())
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

```python
from anydataset.provider.longcat import LongCatViewProvider

ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    dataset_id="my-audio",
    split="train",
).write(dataset, LongCatViewProvider())
```

## 开发

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

设计说明在 [docs/design.md](docs/design.md)，待办事项在 [todo.md](todo.md)。
