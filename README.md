# anydataset

`anydataset` is a PyTorch-first iterable dataset library. It reads samples from one
or more dataset sources and returns formatted `Sample` objects. Multi-source
datasets behave like a single `IterableDataset`; batching and collation stay in
the caller's PyTorch `DataLoader`.

## Install

```bash
pip install -e '.[huggingface,test]'
```

For audio datasets:

```bash
pip install -e '.[huggingface,audio]'
```

## Basic Usage

```python
from torch.utils.data import DataLoader

from anydataset import AnyDataset, Task

dataset = AnyDataset(
    datasets=["mnist:train"],
    task=Task.IMAGE_CLASSIFICATION,
)

for sample in dataset:
    print(sample.dataset_name, sample.sample_index, sample.data.keys())
    break

loader = DataLoader(dataset, batch_size=32, collate_fn=lambda samples: samples)

for batch in loader:
    print(len(batch), batch[0].dataset_name, batch[0].sample_index)
    break
```

`AnyDataset` initialization is intentionally thin. Per-sample formatting
belongs to `formatter`, multi-source iteration order belongs to `strategy`, and
distributed sharding belongs to `.shard(...)`.

```python
rank_dataset = dataset.shard(num_shards=8, shard_id=0)
loader = DataLoader(rank_dataset, batch_size=32, num_workers=4, collate_fn=lambda samples: samples)
```

For multiple sources, choose an iteration strategy at construction time:

```python
from anydataset import RoundRobinStrategy, WeightedRandomStrategy

round_robin = AnyDataset(
    datasets=["fleurs:train", "librispeech_asr:train.100"],
    task=Task.AUDIO_CODEC,
    strategy=RoundRobinStrategy(),
)

weighted = AnyDataset(
    datasets=["fleurs:train", "librispeech_asr:train.100"],
    task=Task.AUDIO_CODEC,
    strategy=WeightedRandomStrategy(
        weights={"fleurs:train": 1.0, "librispeech_asr:train.100": 2.0},
        seed=13,
    ),
)
```

## HuggingFace Streaming

Streaming is enabled through `DatasetSpec.load_options` and is passed to
`datasets.load_dataset(...)`.

```python
from torch.utils.data import DataLoader

from anydataset import AnyDataset, DatasetSpec, Task
from anydataset.tasks import ImageClassificationFormatter

dataset = AnyDataset(
    datasets=["mnist_stream"],
    task=Task.IMAGE_CLASSIFICATION,
    dataset_map={
        "mnist_stream": DatasetSpec(
            source="huggingface",
            path="ylecun/mnist",
            name="mnist_stream",
            split="train",
            load_options={"streaming": True},
        )
    },
    formatter=ImageClassificationFormatter(),
)

loader = DataLoader(dataset, batch_size=32, collate_fn=lambda samples: samples)
```

On Fudan server `145`, direct access to `huggingface.co` timed out during the
streaming smoke test. Use the HuggingFace mirror endpoint there:

```bash
HF_ENDPOINT=https://hf-mirror.com python your_training_script.py
```

The 145 smoke test verified that `load_options={"streaming": True}` returns a
`datasets.iterable_dataset.IterableDataset`, can produce samples through
`AnyDataset`, and can be consumed by PyTorch `DataLoader` without
materializing Arrow files.

## Custom Dataset Map

```python
from anydataset import AnyDataset, DatasetSpec, Task
from anydataset.datasets import LocalFilesDataset
from anydataset.tasks import ImageClassificationFormatter

dataset = AnyDataset(
    datasets=["my_images:train"],
    task=Task.IMAGE_CLASSIFICATION,
    dataset_map={
        "my_images": DatasetSpec(
            source="local_files",
            path="/data/my_images",
            name="my_images",
            adapter=LocalFilesDataset(),
        )
    },
    formatter=ImageClassificationFormatter(),
)
```

Task adapters are registered by dataset name and task:

```python
from anydataset import AnyDataset, DatasetSpec, Task, TaskAdapterRegistry
from anydataset.datasets.local_files.adapters.audio_codec import AudioCodecSampleAdapter

registry = TaskAdapterRegistry()
registry.register(
    "my_audio",
    Task.AUDIO_CODEC,
    lambda spec: AudioCodecSampleAdapter(waveform_key="samples", sample_rate_key="sr"),
)

dataset = AnyDataset(
    datasets=["my_audio:train"],
    task=Task.AUDIO_CODEC,
    dataset_map={
        "my_audio": DatasetSpec(
            source="local_files",
            path="/data/audio.jsonl",
            name="my_audio",
        )
    },
    task_adapter_registry=registry,
)
```

## Development

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

Design decisions live in [docs/design.md](docs/design.md). Pending work lives in
[todo.md](todo.md).
