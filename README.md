# anydataset

`anydataset` is a small PyTorch dataset layer for mapping different physical
sources into one canonical `Sample` shape:

```python
Sample = Mapping[tuple[Role, Modality], AudioItem | ImageItem | TextItem]
```

The source layer only prepares and iterates raw rows. Presets decide how those
rows are parsed into canonical samples.

## Install

```bash
pip install -e '.[huggingface,test]'
```

For audio file loading:

```bash
pip install -e '.[huggingface,audio]'
```

## Presets

Use `Preset` when a built-in dataset already knows both its source and parser.

```python
from anydataset import AudioView, Modality, Preset, Role

dataset = Preset.FLEURS.create(split="validation")
sample = next(iter(dataset))

audio = sample[Role.DEFAULT, Modality.AUDIO]
waveform = audio.views[AudioView.WAVEFORM]
sample_rate = audio.required["sample_rate"]
```

`Preset.spec(...)` returns the physical source only:

```python
from anydataset import Preset

spec = Preset.MNIST.spec(split="train")
```

String shorthands are resolved by `resolve_dataset`:

```python
from anydataset import resolve_dataset

spec = resolve_dataset("mnist:train")
hf = resolve_dataset("hf://ylecun/mnist:train")
unified = resolve_dataset("unified:///data/my_anydataset:train")
```

## Custom Sources

`AnyDataset` is map-style. `IterableAnyDataset` is iterable-style. Both take a
`Spec` and an optional `parse_fn` that maps one raw row to a canonical `Sample`.

```python
from anydataset import AnyDataset, ImageItem, ImageOptKey, ImageView, Modality, Role, Source, Spec

def parse(row):
    return {
        (Role.DEFAULT, Modality.IMAGE): ImageItem(
            views={ImageView.PIXEL: row["image"]},
            optional={ImageOptKey.LABEL: row["label"]},
        )
    }

dataset = AnyDataset(
    Spec(source=Source.HF, path="ylecun/mnist", split="train"),
    parse_fn=parse,
)
```

For local JSON, image, or audio files, use `Source.HF` with Hugging Face
`load_dataset(...)` options such as `data_files` or `data_dir`. For structured
local datasets with canonical samples, use `Source.UNIFIED`.

New physical source types can be registered with a small factory:

```python
from pathlib import Path
from anydataset import IterableAnyDataset, Spec, register_source

class DatabaseSource:
    def prepare(self, spec: Spec, cache_path: Path):
        return connect_rows(spec.path, **spec.load_options)

register_source("database", DatabaseSource)

dataset = IterableAnyDataset(
    Spec(source="database", path="postgresql://host/db", split="train"),
    parse_fn=parse,
)
```

## Multiple Datasets

Combine already-created datasets with `MultipleAnyDataset`.

```python
from anydataset import MultipleAnyDataset, Preset, RoundRobinStrategy

dataset = MultipleAnyDataset(
    [
        Preset.FLEURS.create(split="train"),
        Preset.LIBRISPEECH_ASR.create(split="train.100"),
    ],
    strategy=RoundRobinStrategy(),
)
```

Every dataset exposes `iter_shard(num_shards, shard_id)` for distributed reads.

## Unified Store

`DatasetWriter` writes canonical samples to a self-describing store. The same
store can be read back through `Source.UNIFIED`.

```python
import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioKey,
    AudioView,
    DatasetWriter,
    Modality,
    Role,
    Source,
    Spec,
)

sample = {
    (Role.DEFAULT, Modality.AUDIO): AudioItem(
        views={AudioView.WAVEFORM: torch.zeros(1, 16000)},
        required={AudioKey.SAMPLE_RATE: 16000},
    )
}

DatasetWriter("/data/my_anydataset", dataset_id="toy-audio").write([sample])

dataset = AnyDataset(
    Spec(source=Source.UNIFIED, path="/data/my_anydataset"),
)
restored = dataset[0]
```

`ViewMaterializer` adds derived views to a unified store.

```python
from anydataset import AudioView, Modality, ViewMaterializer, ViewRef

ViewMaterializer(
    input_dir="/data/my_anydataset",
    output_dir="/data/my_anydataset_longcat",
    input_ref=ViewRef(Modality.AUDIO, AudioView.WAVEFORM),
    output_ref=ViewRef(Modality.AUDIO, AudioView.LONGCAT),
    transform=lambda view: {"semantic_codes": view.value.to(torch.int64)},
    provider_name="toy_longcat",
    provider_version="1",
).write()
```

Use `mode="self_contained"` to copy existing views into the output dataset, or
set `input_dir == output_dir` to register the new view in place.

## Development

```bash
/Users/zhuyin/miniconda3/envs/torch2.12/bin/python -m pytest -q
```
