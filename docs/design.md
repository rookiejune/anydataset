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
from anydataset.datasets import HuggingFaceDataset, LocalFilesDataset
from anydataset.tasks import ImageClassificationFormatter

dataset_map = {
    "mnist": DatasetSpec(
        source="huggingface",
        path="ylecun/mnist",
        name="mnist",
        adapter=HuggingFaceDataset(),
    ),
    "my_images": DatasetSpec(
        source="local_files",
        path="/data/my_images",
        name="my_images",
        adapter=LocalFilesDataset(),
    ),
}

dataset = AnyDataset(
    datasets=["mnist:train", "my_images:train"],
    dataset_map=dataset_map,
    task=Task.IMAGE_CLASSIFICATION,
    formatter=ImageClassificationFormatter(),
)

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=lambda samples: samples,
)
```

## 架构边界

- `tasks/<task_name>/` 定义任务的 canonical sample schema 和 per-sample formatter。
- task 只接受 canonical sample，不猜测外部数据集字段名，不维护 alias 表，不处理具体数据集的类别映射。
- `datasets/<dataset_name>/` 定义具体数据集如何下载、缓存、准备和读取 raw item。
- `datasets/<dataset_name>/adapters/<task_name>.py` 把某个数据集的 raw item 映射到某个 task 的 canonical sample。
- `api/` 负责用户入口、resolver、spec、cache、mixing、iteration strategy、sharding、queue 和 dataset/task 组装。
- 顶层 `anydataset.*` 可以保留便捷导出；新实现优先放进上述目录。

## Dataset、Strategy 和 Formatter

- `DatasetSource` 负责单个 resolved `DatasetSpec` 的 prepare/cache、task adapter、per-sample formatter、shard 和 sample iteration；它只接收 `DatasetSpec`，不接收字符串引用或 `dataset_map`。
- `AnyDataset` 持有一个或多个 `DatasetSource`，对外仍表现为普通 `IterableDataset`。
- `AnyDataset` 可以接收 dataset ref/spec 并用外层 `task`、`formatter`、`cache_dir` 初始化 singles；字符串引用只在这一层通过 `DatasetResolver` 解析。
- 如果 `AnyDataset` 接收的是已经实例化的 `DatasetSource`，外层 `dataset_map`、`formatter` 和 `cache_dir` 不再适用。
- `AnyDataset.shard()` 调用每个 child dataset 的 `shard()`。
- `AnyDataset.__iter__` 默认按 `datasets` 声明顺序串行迭代 sample。
- 多数据源迭代顺序由 `IterationStrategy` 决定，例如 `SequentialStrategy`、`RoundRobinStrategy` 和 `WeightedRandomStrategy`。
- `SampleFormatter` 只负责改写单条 `Sample`。
- 采样率、声道数和 clip 截断属于对应 per-sample formatter。
- padding、stack 和 dataclass batch collate 不属于 formatter。

## Resolver 和 Cache

`DatasetSpec` 是 raw dataset 的结构化描述。每个 spec 必须有唯一的
`name`，并且 `dataset_map` 的 key 必须和 `DatasetSpec.name` 一致。

`DatasetResolver` 负责把用户传入的数据集引用解析成 `DatasetSpec`。

默认解析规则：

- `hf://org/name:split` 明确走 HuggingFace。
- `local:///path/to/data:split` 明确走本地文件。
- 普通名字先查 `datasets.catalog.DEFAULT_DATASET_MAP`。
- 查不到就报错，提示用户传入 `dataset_map`。

`CacheManager` 根据 raw dataset 的 source、path、name、split、version、load_options 和 adapter 类型生成稳定缓存路径。cache path 不应因为 task adapter registry 或 sample metadata 不同而分裂。

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
```

第一批通用数据源：

- `HuggingFaceDataset`
- `LocalFilesDataset`

HuggingFace streaming 通过 `DatasetSpec.load_options={"streaming": True}` 显式开启，并透传给 `datasets.load_dataset(...)`。实测在 145 上直连 `huggingface.co` 会超时，使用 `HF_ENDPOINT=https://hf-mirror.com` 可以正常返回 `datasets.iterable_dataset.IterableDataset`，并通过 `AnyDataset` 和 PyTorch `DataLoader` 取样；smoke test 未产生 Arrow materialization 文件。

数据集 adapter 如果支持原生 sharding 且能保留全局 sample index，应优先在 adapter 内实现；API 层的 index modulo sharding 是保底行为。

## Task Adapter Registry

dataset task adapter 负责把 raw item 映射成某个 task 的 canonical sample。它不写进
`DatasetSpec`，而是通过 `TaskAdapterRegistry` 按 `(dataset_name, task)` 注册。

`DatasetSource` 迭代时用 `spec.name` 和 `task` 精确查询 registry；没有注册项时
raw row 原样进入 formatter。内置 adapter 由各 `datasets/<dataset_name>/dataset.py`
里的 `register_task_adapters(registry)` 声明，`default_task_adapter_registry()`
只负责调用这些 dataset registrar。
自定义数据集需要 task adapter 时，应创建 registry 并传给 `AnyDataset` 或
`DatasetSource`。

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

`ImageClassificationFormatter` 负责把单条 sample 的 `image` 转为 channel-first tensor，并把 `label` 转为 `int`。

### Audio Codec

canonical sample 字段：

- `waveform`
- `sample_rate`
- optional `text`

约定：

- 是否包含 `text` 由具体 dataset adapter 决定。
- task 层不维护 music/speech/environment 这类音频来源枚举。
- `AudioCodecFormatter` 只做单条 sample 的 waveform tensor 化、重采样、声道匹配和可选截断，不做 padding 或 batch collate。
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
