# anydataset

`anydataset` is a PyTorch-first iterable dataset library. It reads samples from one
or more dataset sources and returns structured `Sample` objects. Multi-source
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

`datasets` accepts either a single dataset reference such as `"mnist:train"` or a
sequence of references/specs. String references are resolved through the default
catalog and optional `dataset_map`.

`DatasetSpec` describes the physical data source. Dataset adapters are provided
separately through `adapter_map` when raw rows need dataset-specific mapping into
logical modalities.

`AnyDataset` initialization is intentionally thin. Multi-source iteration order
belongs to `strategy`, and distributed sharding belongs to `.shard(...)`.

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

Built-in audio/parquet catalog entries such as `fleurs`, `librispeech_asr`,
`esc50`, and `nsynth` default to `load_options={"streaming": True}`. Explicit
`hf://...` references and custom `DatasetSpec` values do not force streaming;
set `load_options` yourself when you want that behavior.

```python
from torch.utils.data import DataLoader

from anydataset import AnyDataset, DatasetSpec, Task
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
from anydataset.adapters import LocalFilesAdapter

dataset = AnyDataset(
    datasets=["my_images:train"],
    task=Task.IMAGE_CLASSIFICATION,
    dataset_map={
        "my_images": DatasetSpec(
            source="local_files",
            path="/data/my_images",
            name="my_images",
        )
    },
    adapter_map={
        "my_images": LocalFilesAdapter(),
    },
)
```

Dataset adapters can define both row iteration and modality extraction:

```python
from anydataset import AnyDataset, DatasetSpec, Task
from anydataset.adapters import LocalFilesAdapter

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
    adapter_map={
        "my_audio": LocalFilesAdapter(waveform_field="samples", sample_rate_field="sr"),
    },
)
```

## Unified Store

Datasets written by `DatasetWriter` can be read back with `source="unified"`.
The MVP reader supports default-role audio `waveform` and `file` views.

```python
from anydataset import AnyDataset, DatasetSpec, Task

dataset = AnyDataset(
    datasets=DatasetSpec(
        source="unified",
        path="/data/my_anydataset",
        name="my_audio",
        split="train",
    ),
    task=Task.AUDIO_CODEC,
)
```

The same source is available as a string reference:

```python
dataset = AnyDataset(
    datasets="unified:///data/my_anydataset:train",
    task=Task.AUDIO_CODEC,
)
```

`ViewMaterializer` can create a new self-contained unified dataset with an
additional or replaced view:

```python
import torch

from anydataset import AudioView, ModalityKey, ViewMaterializer, ViewRef

ViewMaterializer(
    input_dir="/data/my_anydataset",
    output_dir="/data/my_anydataset_longcat",
    input_ref=ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM),
    output_ref=ViewRef(ModalityKey.AUDIO, AudioView.LONGCAT),
    transform=lambda view: {"semantic_codes": view.value.to(torch.int64)},
    provider_name="toy_longcat",
    provider_version="1",
).write()
```

LongCat can be used as a lazy optional provider. Pass an initialized codec to
avoid importing `anytrain`; otherwise the provider loads
`anytrain.codec.longcat.LongCatAudioCodec` on first use:

```python
from anydataset import LongCatViewProvider

LongCatViewProvider(
    device="cuda",
    n_acoustic_codebooks=2,
    local_files_only=True,
).materializer(
    input_dir="/data/my_anydataset",
    output_dir="/data/my_anydataset_longcat",
).write()
```

## Development

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

Design decisions live in [docs/design.md](docs/design.md). Pending work lives in
[todo.md](todo.md).
