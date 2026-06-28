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
Built-in presets currently include `MNIST`, `CIFAR10`, `FLEURS`,
`LIBRISPEECH_ASR`, `COMMON_VOICE`, `ESC50`, `NSYNTH`, `FSD50K`, and `WMT19`.

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
disk = resolve_dataset("hf-disk:///data/mnist_saved:validation")
store = resolve_dataset("store:///data/my_anydataset:train")
tsv = resolve_dataset("tsv:///data/common_voice/en:train")
csv = resolve_dataset("sharded_csv:///data/bitext:train")
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

Built-in enum sources are `Source.HF`, `Source.HF_DISK`, and `Source.STORE`.
The registry also includes string source keys `tsv` and `sharded_csv`; because
they are registered, they can be used in `Spec(source=...)` and in
`resolve_dataset("<source>://...")` shorthands. `tsv` reads a file path or
`<path>/<split>.tsv`; `sharded_csv` reads `shard_<index>/*.csv`, optionally
under `<path>/<split>/`.

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

def quality_factory():
    return lambda sample: "review" if needs_review(sample) else is_good(sample)


rule = FilterRule("quality_v1_parse_v3_transform_none", quality_factory)

train = FilteredDataset(dataset, rule, labels="accept", device="auto")
audit = FilteredDataset(dataset, rule, labels=("reject", "review"))
```

`True` maps to `"accept"` and `False` maps to `"reject"`. String and enum
labels are stored as their own partitions. The rule `name` is the cache
contract; callers should put predicate, parser, and transform semantics into
`name` when cache reuse should change.

`FilteredDataset` first checks whether the named rule already has a ready cache
for the base dataset. If not, it builds the cache, then exposes only the labels
specified by the caller.

`FilterRule` stores a zero-argument factory, and the factory builds the
predicate inside the process that will execute it. `device="auto"` uses one
spawned process per visible CUDA device and falls back to CPU single-process
execution. Pass `device="cpu"` for explicit single-process CPU filtering, or an
iterable such as `("cpu", "cpu")` or `("cuda:0", "cuda:1")` for explicit
parallel workers. Multi-device filtering launches one fixed worker per device,
sets DDP-style `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT` before calling the factory, and scans an exhaustive runtime-style
index shard so every base sample is covered. Multi-device filtering manages
these environment variables itself; run it as an offline preprocessing step
rather than from inside an existing DDP training process. It uses Python
`spawn`, so factories should be module-level picklable callables.

Partition index files are sharded by `max_shard_samples` (default: 1,000,000),
so large labels do not need one huge parquet file. `commit_samples` (default:
100,000) bounds each in-memory label batch before it is committed to the shard
writer.

Predicates can return `FilterDecision` when a filter should also cache
per-sample JSON metrics:

```python
def metric_factory():
    return lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    )


rule = FilterRule("quality_v2", metric_factory)

result = rule.apply(dataset, metrics=True, device="cpu")
rows = list(result.iter_metrics())
```

Metrics are written under the filter cache and include the original sample
index, normalized label, and metrics payload. Set `metrics=True` explicitly;
when an older partition cache has no metrics side output, the rule is rebuilt.

## Quality Predicates

Quality modules provide reusable predicates for `FilterRule`; they do not own
dataset loading or cache naming.

```python
from anydataset import FilterRule, Preset
from anydataset.quality.translation import Predicate as TranslationQuality

dataset = Preset.WMT19.create(source_lang="zh", target_lang="en")
def translation_factory():
    return TranslationQuality.from_preset(
        Preset.WMT19,
        source_lang="zh",
        target_lang="en",
    )

result = FilterRule("mt_quality_rules_v1_zh_en", translation_factory).apply(
    dataset,
    metrics=True,
)
train = result.select("clean", "usable")
```

`anydataset.quality.translation.Predicate` labels text pairs as `clean`,
`usable`, `review`, or `reject`. The first built-in profile is WMT19 `zh-en`;
other language pairs should pass an explicit `Profile`.

`anydataset.quality.speech.Predicate` scans audio items with same-role text and
labels samples as `accept` or `reject` based on UTMOS, chrF, duration-per-text
unit, peak amplitude, and optional WER/BLEU thresholds:

```python
from anydataset import FilterRule
from anydataset.quality.speech import Predicate as SpeechQuality

def speech_factory():
    return SpeechQuality()

result = FilterRule("speech_quality_v1", speech_factory).apply(
    dataset,
    metrics=True,
)
accepted = result.select("accept")
```

Speech quality warnings such as missing waveform or missing same-role text are
audit signals in the metrics payload; the current predicate rejects only when a
checked audio item fails a configured threshold.

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

`merge()` matches samples by iteration order, not by `sample_id`. The right-hand
dataset must yield samples in the same stable order as the target store for that
merge pass. Do not pass shuffled loaders, runtime-sharded iterables, or
iterators from an active DDP/DataLoader worker context. Delta stores written by
`ViewMaterializer` are safe for this path because part commit restores
`sample_index` order before merge.

For GPU-backed providers, let `devices` control parallelism. `devices="auto"`
uses one spawned worker per visible CUDA device, writes worker logs under
`<output_dir>/logs`, and commits the per-device parts when all workers finish.
Multi-device materialization uses Python `spawn`, so `dataset_factory` and
`provider_factory` must be picklable, module-level callables. Like filtering,
multi-device materialization owns its offline worker processes and should not
be launched from inside an existing DDP training process.
Pass `num_workers` to let each materializer process read samples through a
PyTorch `DataLoader`; this is useful when `parse_fn` does CPU-heavy work such
as file-to-waveform decoding. The materializer sets rank environment variables
for its device workers, and datasets combine rank and DataLoader worker state
inside their runtime shard logic so each sample is covered once.

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatProvider

    return LongCatProvider(device=device)


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    batch_size=8,
    num_workers=4,
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
be derived with `batch.lengths(field_ref)`. When a view or modality materializer
batches a single input reference, `call_batch` may return one output sequence.
When the same batch contains multiple input references, `call_batch` must return
a mapping from `(role, modality)` reference to that reference's output sequence.

`LongCatProvider.call_batch` pads waveform or file input before encoding. If a
batch has multiple audio roles, it encodes each role separately from the same
collated batch. File batches are loaded by the audio provider before padding,
and their effective lengths come from the loaded waveforms because file views do
not carry masks. The current LongCat encoder does not accept masks, so the
provider trims output codes proportionally from each input waveform length
before writing samples to the store.

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

Additional design notes live in `docs/design.md`, filter cache details in
`docs/filter_cache.md`, and quality-filter notes in
`docs/translation_quality.md` and `docs/speech_quality.md`.
