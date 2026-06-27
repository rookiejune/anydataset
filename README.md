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
waveform, sample_rate = audio.views[AudioView.WAVEFORM]
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
store = resolve_dataset("store:///data/my_anydataset:train")
```

## Custom Sources

`AnyDataset` is map-style. `IterableAnyDataset` is iterable-style. Both take a
`Spec` and an optional `parse_fn` that maps one raw row to a canonical `Sample`.

```python
from anydataset import AnyDataset, ImageItem, ImageMeta, ImageView, Modality, Role, Source, Spec

def parse(row):
    return {
        (Role.DEFAULT, Modality.IMAGE): ImageItem(
            views={ImageView.PIXEL: row["image"]},
            meta={ImageMeta.LABEL: row["label"]},
        )
    }

dataset = AnyDataset(
    Spec(source=Source.HF, path="ylecun/mnist", split="train"),
    parse_fn=parse,
)
```

For local JSON, image, or audio files, use `Source.HF` with Hugging Face
`load_dataset(...)` options such as `data_files` or `data_dir`. For structured
local datasets with canonical samples, use `Source.STORE`.

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

## Cached Filter Partitions

`FilterRule` routes a map-style dataset into cached label partitions. The
predicate receives only the sample subset selected by `schema`.

```python
from anydataset import AudioMeta, AudioReq, FilterRule, Modality, Role

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        meta=frozenset({AudioMeta.LABEL}),
    )
}

rule = FilterRule(
    name="quality_v1_parse_v3_transform_none",
    schema=schema,
    predicate=lambda sample: "review" if needs_review(sample) else is_good(sample),
)

result = rule.apply(dataset, num_workers=4)
train = result.select("accept")
audit = result.select("reject", "review")
```

`True` maps to `"accept"` and `False` maps to `"reject"`. String and enum
labels are stored as their own partitions. Rule cache identity includes only
`name` and `schema`; callers should put predicate, parser, and transform
semantics into `name` when cache reuse should change.

Filter cache construction is single-process by default. Pass `num_workers` to
parallelize over map-style index ranges. Partition index files are sharded by
`max_shard_samples` (default: 1,000,000), so large labels do not need one huge
parquet file. `commit_samples` (default: 100,000) bounds each in-memory label
batch before it is committed to the shard writer.

## Store

`DatasetWriter` writes canonical samples to a self-describing store. The same
store can be read back through `Source.STORE`.

```python
import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    DatasetWriter,
    Modality,
    Role,
    Source,
    Spec,
)

sample = {
    (Role.DEFAULT, Modality.AUDIO): AudioItem(
        views={AudioView.WAVEFORM: (torch.zeros(1, 16000), 16000)},
    )
}

DatasetWriter("/data/my_anydataset", dataset_id="toy-audio").write([sample])

dataset = AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset"),
)
restored = dataset[0]
```

Views are stored under `{role}/{modality}/{view}/`; payloads live in that
view directory's `shards/` files. `ViewMaterializer` adds derived views to a
delta store, which can be merged into the base store after it is complete.

```python
from anydataset import AnyDataset, AudioView, Source, Spec, ViewMaterializer
from anydataset.store import read_store_dataset

class ToyLongCat:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, sample_rate = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64)}

dataset = AnyDataset(
    Spec(source=Source.STORE, path="/data/my_anydataset"),
)

delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    dataset_id="toy-audio",
).write(dataset, ToyLongCat())

read_store_dataset("/data/my_anydataset").merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta)))
)
```

## Development

```bash
/Users/zhuyin/miniconda3/envs/torch2.12/bin/python -m pytest -q
```
