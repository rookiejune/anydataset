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
predicate receives the full canonical sample produced by the dataset.

```python
from anydataset.filter import FilterDecision, FilterRule, FilteredDataset

rule = FilterRule(
    name="quality_v1_parse_v3_transform_none",
    predicate=lambda sample: "review" if needs_review(sample) else is_good(sample),
)

train = FilteredDataset(dataset, rule, labels="accept", num_workers=4)
audit = FilteredDataset(dataset, rule, labels=("reject", "review"))
```

`True` maps to `"accept"` and `False` maps to `"reject"`. String and enum
labels are stored as their own partitions. The rule `name` is the cache
contract; callers should put predicate, parser, and transform semantics into
`name` when cache reuse should change.

`FilteredDataset` first checks whether the named rule already has a ready cache
for the base dataset. If not, it builds the cache, then exposes only the labels
specified by the caller.

Filter cache construction is single-process by default. Pass `num_workers` to
parallelize over map-style index ranges. Partition index files are sharded by
`max_shard_samples` (default: 1,000,000), so large labels do not need one huge
parquet file. `commit_samples` (default: 100,000) bounds each in-memory label
batch before it is committed to the shard writer.

Predicates can return `FilterDecision` when a filter should also cache
per-sample JSON metrics:

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

Metrics are written under the filter cache and include the original sample
index, normalized label, and metrics payload. Set `metrics=True` explicitly;
when an older partition cache has no metrics side output, the rule is rebuilt.

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
delta store. Open the store through `Source.STORE` and call `merge()` on the
`AnyDataset` when the delta is complete.

```python
from anydataset import AnyDataset, AudioView, Source, Spec, ViewMaterializer

class ToyLongCat:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, sample_rate = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64)}

def dataset_factory():
    return AnyDataset(Spec(source=Source.STORE, path="/data/my_anydataset"))


def provider_factory(device: str):
    return ToyLongCat()


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="cpu",
)

AnyDataset(Spec(source=Source.STORE, path="/data/my_anydataset")).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta)))
)
```

For GPU-backed providers, let `devices` control parallelism. `devices="auto"`
uses one spawned worker per visible CUDA device, writes worker logs under
`<output_dir>/logs`, and commits the per-device parts when all workers finish.

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatProvider

    return LongCatProvider(device=device)


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)
```

Providers can opt into model-side batching by implementing `call_batch(batch)`
and by passing `batch_size` to the materializer. The `batch` argument is the
same `Batch(sample, masks)` object returned by `collate_fn`; `batch_size=1` or
providers without `call_batch` keep using the per-sample `__call__` path.
`Batch.masks` remains the canonical validity signal, and sequence lengths can
be derived with `batch.lengths(field_ref)`.

`LongCatProvider.call_batch` pads waveform input before encoding. The current
LongCat encoder does not accept masks, so the provider trims output codes
proportionally from the input waveform mask before writing samples to the store.

`ModalityMaterializer` adds a missing modality under the same role. The
provider declares its output view; the materializer infers the output modality
from that view and uses the role's single remaining modality as input. It raises
when the output modality already exists or when the input modality is ambiguous.
Generated items start with empty metadata.

```python
from anydataset import AudioView, ModalityMaterializer, TextView


class ToyTTS:
    output = AudioView.WAVEFORM

    def __call__(self, views):
        text = views[TextView.TEXT]
        return synthesize(text)


delta = ModalityMaterializer(
    output_dir="/data/my_anydataset_tts",
).write(
    dataset_factory=dataset_factory,
    provider_factory=lambda device: ToyTTS(),
    devices="cpu",
)
```

Built-in providers follow the model/backend name, for example
`MossTTSProvider` for text-to-audio and `WhisperASRProvider` for
audio-to-text.

## Development

```bash
/Users/zhuyin/miniconda3/envs/torch2.12/bin/python -m pytest -q
```
