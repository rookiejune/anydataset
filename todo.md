# TODO

## 目标

实现一个 PyTorch-first 的 iterable dataset。

它支持传入一些数据集的名字，用缓存的方式下载到硬盘，然后迭代式加载。核心特性是：内部可以组织多个不同数据集，并且按照 task 对应的 batch dataclass 定义返回训练可用的 PyTorch batch。

同一个 dataset 可以作为多种 task 的来源，所以 task 应该是独立 module/package，而不是写死在 dataset 或 adapter 里。dataset 层只负责解析、缓存、加载和混合样本；tasks 层负责把统一 sample 转换成具体任务的 batch。

## 拍板方向

- 主要面向 PyTorch 训练。
- `torch` 是必需安装依赖，不作为 optional extra。
- 主入口应该继承或兼容 `torch.utils.data.IterableDataset`。
- batch 输出默认是 PyTorch tensor。
- 返回值一定是 dataclass。
- batch dataclass 的定义由 task 决定，不单独提供 `batch_format="dataclass"` 这种配置。
- 例如 `Task.IMAGE_CLASSIFICATION` 固定返回 `ImageClassificationBatch`，其中包含 image tensor、label tensor 和 meta。
- meta 记录每条样本来自哪个数据集、是该数据集里的第几个样本。
- batch dataclass 里的字段应当按 batch 维度组织；tensor 字段第一维是 batch size，meta 里的每个 key 是长度等于 batch size 的 list。
- 数据集来源通过一个可自定义的 map/registry 管理。
- 默认数据集解析可以根据 dataset name 自动选择合适的数据源。
- 多数据集混合默认使用 `weighted`。
- 不做断点续训和 `state_dict()` / `load_state_dict()`，iterable dataset 本来就不强保证这些。

## 示例 API

```python
from anydatasets import AnyIterableDataset, Task

dataset = AnyIterableDataset(
    datasets=[
        "mnist:train",
        "cifar10:train",
    ],
    task=Task.IMAGE_CLASSIFICATION,
    batch_size=32,
    weights={
        "mnist:train": 1.0,
        "cifar10:train": 2.0,
    },
    cache_dir="~/.cache/anydatasets",
)

for batch in dataset:
    # batch 是 ImageClassificationBatch dataclass，例如：
    # batch.images: torch.Tensor
    # batch.labels: torch.Tensor
    # batch.meta.dataset_names: list[str]
    # batch.meta.sample_indices: list[int]
    ...
```

自定义 registry：

```python
from anydatasets import AnyIterableDataset, DatasetSpec, Task
from anydatasets.adapters import HuggingFaceAdapter, LocalFilesAdapter

dataset_map = {
    "mnist": DatasetSpec(
        source="huggingface",
        path="ylecun/mnist",
        adapter=HuggingFaceAdapter(),
    ),
    "my_images": DatasetSpec(
        source="local_files",
        path="/data/my_images",
        adapter=LocalFilesAdapter(),
    ),
}

dataset = AnyIterableDataset(
    datasets=["mnist:train", "my_images:train"],
    dataset_map=dataset_map,
    task=Task.IMAGE_CLASSIFICATION,
    batch_size=32,
)
```

batch schema 由 `task` 决定。第一版不支持用户传入自定义 collate，也不支持用户自定义 batch dataclass。

## 核心抽象

### 1. AnyIterableDataset

主入口，负责串起 registry、cache、adapter、mixer、batch builder。

建议行为：

- 继承 `torch.utils.data.IterableDataset`。
- `__iter__()` 内部创建样本流。
- 每次迭代返回一个 batch，而不是单条 sample。
- 所有 batch 默认应当是 dataclass，字段中包含 PyTorch tensor 和按样本对齐的 meta list。

构造参数草案：

```python
AnyIterableDataset(
    datasets,
    task,
    batch_size,
    dataset_map=None,
    weights=None,
    cache_dir="~/.cache/anydatasets",
    shuffle=True,
    seed=None,
    drop_last=False,
)
```

### 2. DatasetRegistry

负责把用户传入的数据集名字解析成内部 `DatasetSpec`。

职责：

- 支持用户传入 `dataset_map`。
- 如果用户没有提供，则使用默认 registry。
- 根据 dataset name 推断数据源。
- 解析 `"name:split"` 这种格式。
- 找到对应 adapter。

建议数据结构：

```python
DatasetSpec(
    name="mnist",
    split="train",
    source="huggingface",
    path="ylecun/mnist",
    version=None,
    adapter=HuggingFaceAdapter(),
    options={},
)
```

默认解析规则可以先做得保守：

- `hf://org/name:split` 明确走 HuggingFace。
- `local:///path/to/data:split` 明确走本地文件。
- 普通名字先查内置 `DEFAULT_DATASET_MAP`。
- 查不到就报错，提示用户传入 `dataset_map`。

### 3. CacheManager

负责缓存、下载和校验。

职责：

- 根据 dataset spec 生成稳定缓存路径。
- 检查本地是否已经准备好。
- 调用 adapter 的 prepare/download 逻辑。
- 写入 metadata。

建议缓存结构：

```text
cache_dir/
  source/
    dataset_name/
      version_or_hash/
        raw/
        processed/
        metadata.json
```

### 4. DatasetAdapter

每种数据源一个 adapter，统一转成 sample dict。

统一接口：

```python
class DatasetAdapter:
    def prepare(self, spec, cache_dir):
        ...

    def iter_samples(self, manifest):
        ...
```

第一版建议支持：

- `HuggingFaceAdapter`
- `LocalFilesAdapter`

后续扩展：

- `TorchvisionAdapter`
- `WebDatasetAdapter`
- 用户自定义 adapter

### 5. tasks module / Task / BatchBuilder

这是这个库最关键的抽象。

task 使用枚举类，而不是裸字符串。枚举项尽量使用 `auto()`，避免手写重复字符串。

```python
from enum import Enum, auto

class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Task(AutoNameEnum):
    IMAGE_CLASSIFICATION = auto()
```

这样保留了 `auto()`，同时 `Task.IMAGE_CLASSIFICATION.value` 会稳定是 `"image_classification"`。

task 负责定义：

- 单条 sample 需要哪些字段。
- 不同数据集字段如何归一化。
- 如何把 Python 值、PIL image、numpy array 等转换成 torch tensor。
- 一个 batch 应该长什么样，也就是返回哪个 dataclass。
- 枚举值到 batch dataclass 的固定映射。

任务代码应放在独立 `tasks/` module 中。这样同一个数据集可以根据不同 `Task` 枚举，走不同的 task builder。

内置 task 示例：

```python
ImageClassificationTask(
    image_key="image",
    label_key="label",
)
```

内置 batch dataclass 示例：

```python
from dataclasses import dataclass

@dataclass
class BatchMeta:
    dataset_names: list[str]
    sample_indices: list[int]

@dataclass
class ImageClassificationBatch:
    images: torch.Tensor
    labels: torch.Tensor
    meta: BatchMeta
```

字段约定：

- `images`: `[batch, channels, height, width]` 的 `torch.Tensor`。
- `labels`: `[batch]` 的 `torch.Tensor`。
- `meta.dataset_names`: 长度为 batch size 的 `list[str]`。
- `meta.sample_indices`: 长度为 batch size 的 `list[int]`。

task 第一版只支持 `Task` 枚举，例如 `Task.IMAGE_CLASSIFICATION`。后续可以增加更多枚举项和对应的内置 batch dataclass，但不开放用户自定义 batch schema。

### 6. WeightedDatasetMixer

负责从多个数据集 sample iterator 中按权重采样。

默认策略：

- 如果用户传入 `weights`，按权重采样。
- 如果没传，所有数据集等权重。
- 某个数据集耗尽后，从采样池里移除。
- 所有数据集耗尽后结束。

后续可选策略：

- `concat`
- `round_robin`
- 无限重复采样

## MVP 范围

第一版建议做：

- 创建标准 Python 包结构。
- `AnyIterableDataset` 继承 `torch.utils.data.IterableDataset`。
- 支持传入多个 dataset name。
- 支持 `dataset_map` 自定义数据源。
- 提供一个很小的 `DEFAULT_DATASET_MAP`。
- 支持 `hf://` 和 `local://` 这类显式数据源前缀。
- 支持缓存目录。
- 支持 `HuggingFaceAdapter`。
- 支持 `LocalFilesAdapter` 的最小版本。
- 支持 `task=Task.IMAGE_CLASSIFICATION`。
- `task=Task.IMAGE_CLASSIFICATION` 返回 `ImageClassificationBatch` dataclass。
- 不支持用户传入 callable 作为 task/collate。
- 默认按 `weighted` 混合多个数据集。
- 不实现断点续训。
- 写基础单元测试覆盖 registry、cache path、weighted mixer、task enum、image classification batch builder。

建议目录：

```text
anydatasets/
  pyproject.toml
  src/
    anydatasets/
      __init__.py
      dataset.py
      registry.py
      cache.py
      mixing.py
      tasks/
        __init__.py
        base.py
        image_classification.py
      adapters/
        __init__.py
        base.py
        huggingface.py
        local_files.py
  tests/
    test_registry.py
    test_cache.py
    test_mixing.py
    test_tasks.py
```

## 已确认的 image_classification batch

第一版固定为：

- `ImageClassificationBatch.images`
- `ImageClassificationBatch.labels`
- `ImageClassificationBatch.meta.dataset_names`
- `ImageClassificationBatch.meta.sample_indices`

如果后续需要兼容其他模型生态，可以增加新的内置 task 或 task 变体，例如返回 `HFImageClassificationBatch`，但仍然由 task 决定 dataclass 定义。
