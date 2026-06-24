# anydataset Design

本文档记录已经定下来的设计。还没有拍板的事项留在 `todo.md`。

## 目标

`anydataset` 是一个 PyTorch-first 的 iterable dataset 库。它从一个或多个数据源迭代读取样本，对外表现为普通 `torch.utils.data.IterableDataset`。

基本方向：

- `torch` 是必需依赖。
- 主入口继承或兼容 `torch.utils.data.IterableDataset`。
- `AnyDataset` 默认迭代返回 unbatched `Sample`。
- 不提供内部 `.dataloader()`；模型侧如果需要 tensor batch，应在项目自己的 PyTorch `DataLoader` / `collate_fn` 中处理。
- 不实现断点续训，也不承诺 `state_dict()` / `load_state_dict()`。

## 公开入口

基础用法：

```python
from torch.utils.data import DataLoader

from anydataset import AnyDataset, Task

dataset = AnyDataset(
    datasets=["mnist:train"],
    task=Task.IMAGE_CLASSIFICATION,
    cache_dir="~/.cache/anydataset",
)

for sample in dataset:
    # sample.data 包含该数据源的样本字段
    ...

loader = DataLoader(dataset, batch_size=32, collate_fn=lambda samples: samples)

for batch in loader:
    # batch: list[Sample]
    ...
```

自定义 dataset map：

```python
from anydataset import AnyDataset, DatasetSpec, Task
from anydataset.adapters import HuggingFaceAdapter, LocalFilesAdapter

dataset_map = {
    "mnist": DatasetSpec(
        source="huggingface",
        path="ylecun/mnist",
        name="mnist",
    ),
    "my_images": DatasetSpec(
        source="local_files",
        path="/data/my_images",
        name="my_images",
    ),
}
adapter_map = {
    "mnist": HuggingFaceAdapter(),
    "my_images": LocalFilesAdapter(),
}

dataset = AnyDataset(
    datasets=["mnist:train", "my_images:train"],
    dataset_map=dataset_map,
    adapter_map=adapter_map,
    task=Task.IMAGE_CLASSIFICATION,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=lambda samples: samples,
)
```

## 架构边界

- `tasks/<task_name>/` 定义任务的 canonical sample schema，并调用 dataset adapter 的模态方法组装任务需要的 sample。
- task 不猜测外部数据集字段名，不维护 alias 表，不处理具体数据集的类别映射。
- `adapters/<dataset_name>.py` 定义具体数据集如何 prepare/cache、读取 raw item，以及如何从 raw item 提供统一模态抽象，例如 `audio(row, role=None)`、`text(row, role=None)`。
- `adapters/catalog.py` 保留内置 spec/catalog；不再维护旧 `datasets/` 包。
- `api/` 负责用户入口、resolver、spec、cache、mixing、iteration strategy、sharding、queue 和 dataset/task 组装。
- 顶层 `anydataset.*` 可以保留便捷导出；新实现优先放进上述目录。

## Dataset、Strategy 和 Adapter

- `DatasetSource` 负责单个 resolved `DatasetSpec` 的 prepare/cache、dataset adapter、task adapter、shard 和 sample iteration；它只接收 `DatasetSpec`，不接收字符串引用或 `dataset_map`。
- `AnyDataset` 持有一个或多个 `DatasetSource`，对外仍表现为普通 `IterableDataset`。
- `AnyDataset` 可以接收单个 dataset ref/spec，或多个 dataset ref/spec，并用外层 `task`、`cache_dir` 和 `adapter_map` 初始化 singles；字符串引用只在这一层通过 resolver helper 解析。
- 如果 `AnyDataset` 接收的是已经实例化的 `DatasetSource`，外层 `dataset_map`、`adapter_map` 和 `cache_dir` 不再适用。
- `AnyDataset.shard()` 调用每个 child dataset 的 `shard()`。
- `AnyDataset.__iter__` 默认按 `datasets` 声明顺序串行迭代 sample。
- 多数据源迭代顺序由 `IterationStrategy` 决定，例如 `SequentialStrategy`、`RoundRobinStrategy` 和 `WeightedRandomStrategy`。
- 采样率、声道数、clip 截断等同一模态内规格统一不属于 dataset adapter；后续放到 normalizer / transform / pipeline 侧。
- padding、stack 和 dataclass batch collate 不属于 dataset。

## Resolver 和 Cache

`DatasetSpec` 是 raw dataset 的物理描述，只表达 source、path、name、split、version 和 load options。每个 spec 必须有唯一的 `name`，并且 `dataset_map` 的 key 必须和 `DatasetSpec.name` 一致。

Dataset adapter 不属于 `DatasetSpec`。内置数据集的 adapter 由 `adapters.catalog.DEFAULT_ADAPTER_MAP` 提供；自定义 raw dataset 需要特殊字段映射时，通过 `AnyDataset(adapter_map={name: adapter_or_factory})` 或直接构造 `DatasetSource(adapter=...)` 传入。

`DatasetResolver` 负责把用户传入的数据集引用解析成 `DatasetSpec`。公开 helper
`resolve_dataset_spec` / `resolve_dataset_specs` 提供兼容层，把单个字符串当成一个
dataset ref，而不是字符序列。

默认解析规则：

- `hf://org/name:split` 明确走 HuggingFace。
- `local:///path/to/data:split` 明确走本地文件。
- `unified:///path/to/dataset:split` 明确走统一格式目录。
- 普通名字先查 `adapters.catalog.DEFAULT_DATASET_MAP`。
- 查不到就报错，提示用户传入 `dataset_map`。

`CacheManager` 根据 raw dataset 的 source、path、name、split、version 和 load_options 生成稳定缓存路径。cache path 不应因为 adapter 或任务侧 sample 解释方式不同而分裂。

首次 materialize 共享 cache 时必须用跨进程文件锁保护，只有一个进程执行下载或准备，其它进程等待 `.ready` 后再读取。

## Dataset Adapter

统一接口：

```python
class DatasetAdapter:
    def prepare(self, spec, cache):
        ...

    def iter_samples(self, manifest):
        ...

    def iter_indexed_samples(self, manifest, num_shards=1, shard_id=0):
        ...

    def audio(self, row, role=None):
        ...

    def text(self, row, role=None):
        ...
```

第一批通用数据源：

- `HuggingFaceAdapter`
- `LocalFilesAdapter`
- `UnifiedDatasetAdapter`

HuggingFace streaming 通过 `DatasetSpec.load_options={"streaming": True}` 显式开启，并透传给 `datasets.load_dataset(...)`。内置大音频/parquet 数据集 `fleurs`、`librispeech_asr`、`esc50` 和 `nsynth` 默认开启 streaming；通用 `hf://...` 引用和用户传入的 `dataset_map` / `DatasetSpec` 不强行开启 streaming，由调用方显式设置。实测在 145 上直连 `huggingface.co` 会超时，使用 `HF_ENDPOINT=https://hf-mirror.com` 可以正常返回 `datasets.iterable_dataset.IterableDataset`，并通过 `AnyDataset` 和 PyTorch `DataLoader` 取样；smoke test 未产生 Arrow materialization 文件。

`UnifiedDatasetAdapter` 读取 `DatasetWriter` 生成的统一格式目录，要求 dataset 和每个读取的 view 都已 `.ready`。MVP 支持默认 role 的 `audio.views.waveform` 和 `audio.views.file`：waveform 从 shard payload 加载为 tensor，file view 解包到 cache 并以本地路径返回。

`ViewMaterializer` 从已有统一格式 dataset 读取一个 input view，调用显式传入的 provider/transform 生成 output view，并写到新的自包含统一格式目录。输出目录会复制保留原有 view，并把新 view 的 revision 写进 `dataset.json`；revision 由 provider name、provider version、config、input view 和 output view 决定。

`LongCatViewProvider` 是 `ViewMaterializer` 的一个具体 provider：默认从 `audio.views.waveform` 读取 waveform，要求 sample manifest 带 `sample_rate`，输出 `audio.views.longcat`，payload 包含 `semantic_codes`、`acoustic_codes` 和输入采样率。LongCat 依赖保持 lazy optional；调用方可以传入已有 codec，未传时才导入 `anytrain.codec.longcat.LongCatAudioCodec`。

数据集 adapter 如果支持原生 sharding 且能保留全局 sample index，应优先在 adapter 内实现；API 层的 index modulo sharding 是保底行为。

自定义数据集需要模态抽取时，应在 `AnyDataset.adapter_map` 或 `DatasetSource.adapter` 中传入完整的 dataset adapter。这个对象同时负责 `prepare(...)`、`iter_samples(...)` 和 `audio(...)` / `text(...)` 等模态方法。

## Tasks

`Task` 使用 `auto()` 生成稳定字符串值：

```python
class Task(AutoNameEnum):
    IMAGE_CLASSIFICATION = auto()
    AUDIO_CODEC = auto()
```

task 固定映射到对应 canonical sample schema。dataset 只返回单条 `Sample`。

### Image Classification

canonical sample 字段：

- `image`
- `label`

图像 tensor 化、channel 顺序统一和 label dtype 统一不放在 dataset adapter 主链路中；训练侧通过自己的 normalizer、transform 或 collate 处理。

### Audio Codec

canonical sample 字段：

- `audio.sample_rate`
- `audio.views`
- optional `audio.duration`
- optional `audio.label`
- optional `audio.labels`
- `text.content`
- optional `text.lang`

约定：

- 音频表示放在 `audio.views`，例如 `waveform`、`file`、`longcat` 和 `dac`。
- `audio.views` 只要求至少有一个可用 view；具体训练用哪个 view 由 normalizer、model adapter 或上层 collate 侧决定。
- 文件路径也是一种 audio view，使用 `audio.views.file` 表达；读取后应按 `audio.sample_rate` 对齐。
- `audio.sample_rate` 表达当前 audio sample 逻辑结构中定义的采样率，不再区分 `source_sample_rate`。
- 音频类别放在 `audio.label` 或 `audio.labels`。
- 文本模态放在 `text` 下，正文是 `text.content`，语言是 `text.lang`。
- speech 是 `audio` + `text` 的组合，不单独引入 `annotations` 容器。
- task 层不维护 music/speech/environment 这类音频来源枚举。
- `anydataset.modalities` 定义顶层模态 key，例如 `ModalityKey.AUDIO` 和 `ModalityKey.TEXT`。
- `anydataset.modalities.audio` 定义基础音频模态：`AudioKey` 表达必选字段 `views` 和 `sample_rate`，`AudioOptKey` 表达可选字段 `duration`、`label` 和 `labels`，`AudioView` 表达通用音频 view，例如 `waveform`、`file`、`longcat` 和 `dac`。
- `anydataset.modalities.text` 定义基础文本模态内部字段：`TextKey` 表达必选字段 `content`，`TextOptKey` 表达可选字段 `lang`。
- `tasks.audio_codec.schema` 只组合基础 audio/text schema，不额外定义 speech annotation key。
- 这些 key / view 都使用 Python `StrEnum` + `auto()` 定义，代码直接用 enum member 作为 dict key。
- `AudioCodecAdapter` 负责调用 dataset adapter 的 `audio(row)` 和可选 `text(row)`，组装 audio codec task 需要的 canonical sample。
- 数据集字段名、嵌套 audio 对象和转录文本字段名属于对应 dataset adapter。

第一批内置音频数据集：

- `fleurs`
- `librispeech_asr`
- `esc50`
- `nsynth`
- `fsd50k`

只保留这些规范名字，不额外增加短 alias。

## Queue 和 Sharding

- 大数据集不应在训练迭代路径上同步阻塞所有数据源。
- 单个慢数据源未就绪时，不应阻塞其它已有样本的数据源。
- 如果需要避免慢数据源阻塞，应实现 prefetching iteration strategy，训练侧优先消费已准备好的 `Sample`。
- 如果所有 producer 都没有产出，或训练长期快于数据生产，消费侧仍然需要等待或超时失败，不能伪造 batch。

`IterableDataset` 在 PyTorch `DataLoader(num_workers>0)` 和 DDP 多进程下会被复制。`anydataset` 的约定是：

- DDP / 多 rank 训练前必须显式调用 `.shard(num_shards, shard_id)`。
- dataset 检测到 DDP / 多 rank 但没有显式 shard 时会报错。
- 普通 DataLoader worker 不要求用户显式 `.shard()`；`DatasetSource` 会按 worker 切分，避免 worker 间重复。
- PyTorch `DataLoader` 的 `batch_size` 和 `collate_fn` 由外部训练代码控制。
- DDP 训练中各 rank 的 batch 数可能因有限数据源分片不均而不同；需要严格同步步数时应使用 `drop_last=True` 或上层固定 `steps_per_epoch`。

## FSD50K

FSD50K 使用 HF repo 中的单个 wav 文件清单，按 shard 逐个下载。FMA-small 这种大 zip 数据集不要伪装成 streaming，除非先有稳定的分片来源。
