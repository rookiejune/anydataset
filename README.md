# anydataset

[简体中文](README.zh-CN.md)

`anydataset` is a small PyTorch dataset layer for mapping different physical
sources into one canonical `Sample` shape:

```python
Sample = Mapping[tuple[Role, Modality], AudioItem | ImageItem | TextItem]
```

The source layer only prepares and iterates raw rows. Presets decide how those
rows are parsed into canonical samples.

## Install

```bash
pip install anydataset
```

For Hugging Face datasets or audio file loading:

```bash
pip install 'anydataset[huggingface,audio]'
```

For local development:

```bash
pip install -e '.[huggingface,audio,dev]'
```

Model-backed providers are optional and come from `anytrain`. Install the
matching extra before using them: `anytrain[longcat]` for `LongCatProvider`,
`anytrain[speech]` for `WhisperASRProvider` and the default speech-quality
evaluator, or `anytrain[moss-tts]` plus `anydataset[audio]` for
`MossTTSProvider`.

## Presets

Use `AnyDataset.preset()` or `IterableAnyDataset.preset()` when a built-in
dataset already knows both its source and parser.
Map-style presets are `MNIST`, `CIFAR10`, and `FSD50K`. Iterable presets are
`FLEURS`, `LIBRISPEECH_ASR`, `COMMON_VOICE`, `ESC50`, `NSYNTH`, and `WMT19`.
Using a preset through the wrong dataset type raises an error. Both constructors
accept `transforms=...` for item-level transforms after parsing.

```python
from anydataset import IterableAnyDataset
from anydataset.types import AudioView, Modality, Role

dataset = IterableAnyDataset.preset("fleurs", split="validation")
sample = next(iter(dataset))

audio = sample[Role.DEFAULT, Modality.AUDIO]
waveform, sample_rate = audio.views[AudioView.WAVEFORM]
```

`Preset.spec(...)` returns the physical source only:

```python
from anydataset import Preset

spec = Preset.MNIST.spec(split="train")
```

`FSD50K` is map-style and accepts only an optional Hugging Face `revision`.
The revision defaults to `main` and is used for file discovery, payload
downloads, and the source cache identity.

```python
from anydataset import AnyDataset

fsd50k = AnyDataset.preset(
    "fsd50k",
    split="dev",
    revision="refs/convert/parquet",
)
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
from anydataset import AnyDataset, Source, Spec
from anydataset.types import (
    ImageItem,
    ImageMeta,
    ImageView,
    Modality,
    Role,
)

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
`resolve_dataset("<source>://...")` shorthands. `tsv` reads a file path,
`<path>/<split>.tsv`, or the same split under ordered `subdirs` load options;
`sharded_csv` reads numeric CSV files under
`shard_<index>/<number>.csv`, optionally under `<path>/<split>/`. Non-numeric
CSV file names are ignored and logged as warnings.

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

Iterable sources that can select rows without scanning the full stream may also
implement `iter_indexed_shard(dataset, *, num_shards, shard_id)`. This source
method must yield `(sample_index, row)` tuples for the exact dense global modulo
shard: indexes start at `shard_id` and advance by `num_shards`. Anydataset
validates tuple shape and index progression before filter or materializer code
sees the rows; the source remains responsible for complete coverage and the
row-to-index association. A raw dataset `shard()` or `iter_indexed_shard()`
method alone is not sufficient, because a locally enumerated native shard does
not preserve global indexes.

The built-in `hf-disk`, `store`, and `sharded_csv` sources provide this indexed
path through random access. TSV and Hugging Face streaming datasets use the
full-stream modulo fallback because their public shard APIs do not propagate
original global row indexes.

Caches are rooted at `ANYDATASET_HOME`, or `~/.cache/anydataset` when the
environment variable is unset. Source prepare caches live under
`$ANYDATASET_HOME/cache/sources/<spec_id>`, and filter partitions live under
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`.
Runtime warnings and worker logs live under
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`.

Every physical `Spec` field participates in `Spec.id`: `source`, `path`,
`split`, `version`, and `load_options`. Change `version` or a load option when
the same path denotes a different physical snapshot; source prepare caches are
reused only for the resulting identity.

The built-in `sharded_csv` source keeps CSV files as the readable source of
truth and prepares one Parquet cache part per CSV file under
`$ANYDATASET_HOME/cache/sources`. Preparation converts changed files in a
spawned process pool and atomically commits the cache manifest. Dataset reads
then use Parquet row groups for map-style random access.

## Multiple Datasets

Combine already-created datasets with `MultipleAnyDataset`.

```python
from anydataset import IterableAnyDataset, MultipleAnyDataset
from anydataset.dataset import RoundRobinStrategy

dataset = MultipleAnyDataset(
    [
        IterableAnyDataset.preset("fleurs", split="train"),
        IterableAnyDataset.preset("librispeech_asr", split="train.100"),
    ],
    strategy=RoundRobinStrategy(),
)
```

Every dataset exposes `iter_shard(num_shards, shard_id)` for distributed reads.
`MultipleAnyDataset` itself is not a filter cache identity; filter or cache the
child datasets before combining them.

The default `SequentialStrategy` exhausts each child in order.
`RoundRobinStrategy` interleaves active children evenly, while
`WeightedRandomStrategy(weights=..., seed=...)` uses weights to choose the next
active child. It changes interleaving order rather than resampling: every child
with a positive weight is still exhausted.

## DataLoader Schemas

`Schema` maps each `(Role, Modality)` reference to the views and metadata that
a training batch needs. `collate_fn(schema)` selects those fields and returns a
`Batch`; it does not fill in missing fields implicitly.

```python
from torch.utils.data import DataLoader

from anydataset.dataset import collate_fn
from anydataset.types import AudioReq, AudioView, Modality, Role

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}

loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    collate_fn=collate_fn(schema),
)
batch = next(iter(loader))
```

Use roles to distinguish multiple items with the same modality. For example, a
machine translation schema can request source and target text independently:

```python
from anydataset.types import Modality, Role, TextReq, TextView

text = TextReq(views=frozenset({TextView.TEXT}))
schema = {
    (Role.SOURCE, Modality.TEXT): text,
    (Role.TARGET, Modality.TEXT): text,
}
```

When a built-in task already describes the required fields, use its default
schema or collator directly:

```python
from anydataset import Task

schema = Task.AUDIO_CODEC.schema()
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=Task.AUDIO_CODEC.collate_fn(),
)
```

`Batch.sample` has the same logical structure as one `Sample`, with each field
batched. Generic tensors with matching shapes are stacked; when only their last
dimension varies, that dimension is padded and recorded in `Batch.masks`.
Waveforms are first converted with `torch.as_tensor` and use the same rule.
Codec views have the stricter per-sample shape `[frame, codebook]`; the frame
axis is padded, producing `[batch, frame, codebook]` and a `[batch, frame]`
mask. Mapping views are collated recursively and must have consistent keys and
sequence lengths. Other values are returned as lists.

```python
from anydataset.dataset import FieldGroup, FieldRef

audio_ref = (Role.DEFAULT, Modality.AUDIO)
waveform, sample_rate = batch.sample[audio_ref].views[AudioView.WAVEFORM]
waveform_mask = batch.masks[
    FieldRef(
        ref=audio_ref,
        group=FieldGroup.VIEWS,
        key=AudioView.WAVEFORM,
    )
]
```

Schema fields must exist in every sample in the batch. Convert values to
tensors and normalize dtype or device in the preset parser or dataset
transforms, before collation.

## Cached Filter Partitions

`FilterRule` routes a map-style dataset into cached label partitions. The
predicate receives the full canonical sample produced by the dataset.

```python
from anydataset.filter import FilterDecision, FilteredDataset, FilterRule

def quality_factory():
    return lambda sample: "review" if needs_review(sample) else is_good(sample)


def dataset_factory():
    return build_dataset()


filtered = FilteredDataset(
    "quality_v1_parse_v3_transform_none",
    quality_factory,
    dataset_factory=dataset_factory,
    device="cpu",
)
train = filtered.select_by("accept")
audit = filtered.select_by("reject", "review")

rule = FilterRule("quality_v1_parse_v3_transform_none", quality_factory)
again = rule.apply(dataset_factory=dataset_factory, labels="accept", device="cpu")
```

`True` maps to `"accept"` and `False` maps to `"reject"`. String and enum
labels are stored as their own partitions. The rule `name` is the cache
contract; callers should put predicate, parser, and transform semantics into
`name` when cache reuse should change.

`FilteredDataset(...)` checks whether the named rule already has a ready cache
for the base dataset. If not, it builds the cache. It selects every available
label by default. Use `select_by(...)` to derive a label view over the same
cache. `FilterRule.apply(...)` is a convenience wrapper that forwards its
`name` and `factory` to `FilteredDataset`.

Filter cache identity is automatic for physical datasets and library-owned
merged children. If a merged dataset contains an external map-style child such
as a list or application dataset, pass a non-empty `input_id` to `apply()` or
`FilteredDataset(...)`. The ID versions the entire input snapshot and augments
the automatic class, `Spec`, and sample-count identity. Change it when external
content or ordering changes; `FilterRule.name` continues to version predicate
semantics. The ID is preserved by the filtered `dataset_factory`, pickle, and
chained filters.

`FilterRule` stores a zero-argument factory, and the factory builds the
predicate inside the process that will execute it. `device="auto"` resolves to
all visible CUDA devices, or to CPU when CUDA is unavailable. One resolved
device runs in the calling process; more than one launches a fixed worker per
device with `Runtime.process_start_method` (`"spawn"` by default). Pass
`device="cpu"` for explicit CPU execution, or an iterable such as
`("cpu", "cpu")` or `("cuda:0", "cuda:1")` for explicit parallel workers.
Multi-device filtering
sets DDP-style `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT` before calling factories, and scans an exhaustive runtime-style
index shard so every base sample is covered. Multi-device filtering manages
these environment variables itself; run it as an offline preprocessing step
rather than from inside an existing DDP training process. It uses Python
`spawn`, so the dataset entry point is always `dataset_factory=...`. Both the
dataset factory and predicate factory should be module-level picklable
callables.
Pass `num_workers` to let each execution process read samples through a PyTorch
`DataLoader`; `batch_size` controls the loader batch size.

Partition index files are sharded by `max_shard_samples` (default: 1,000,000),
so large labels do not need one huge parquet file. `commit_samples` (default:
100,000) bounds each in-memory label batch before it is committed to the shard
writer. Filter cache construction uses hidden resume fragments and replays them
into the final cache when all samples are covered. `write_workers` defaults to
one background writer so predicate execution can overlap with parquet writes;
`write_prefetch` bounds pending write jobs.

Predicates can return `FilterDecision` when a filter should also cache
per-sample JSON metrics:

```python
def metric_factory():
    return lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    )


rule = FilterRule("quality_v2", metric_factory)

filtered = rule.apply(dataset_factory=dataset_factory, metrics=True, device="cpu")
rows = list(filtered.iter_metrics())
```

Metrics are written under the filter cache and include the original sample
index, normalized label, and metrics payload. Set `metrics=True` explicitly;
when an older partition cache has no metrics side output, the rule is rebuilt.
Completed caches are immutable generations with reader leases. See
[`docs/filter_cache.md`](docs/filter_cache.md) for their cleanup contract and
the `cleanup_filter_generations(...)` API.

## Quality Predicates

Quality modules provide reusable predicates for `FilterRule`; they do not own
dataset loading or cache naming.

`FilterRule` accepts map-style inputs. Because WMT19 is an iterable preset,
materialize it once to a store before filtering it:

```python
from anydataset import AnyDataset, FilterRule, IterableAnyDataset, Preset, Source, Spec
from anydataset.quality.translation import Predicate as TranslationQuality

source = IterableAnyDataset.preset(
    "wmt19", source_lang="zh", target_lang="en"
)
source.write("/data/wmt19-zh-en", dataset_id="wmt19-zh-en", split="train")


def dataset_factory():
    return AnyDataset(
        Spec(source=Source.STORE, path="/data/wmt19-zh-en", split="train")
    )


def translation_factory():
    return TranslationQuality.from_preset(
        Preset.WMT19,
        source_lang="zh",
        target_lang="en",
    )

filtered = FilterRule("mt_quality_rules_v1_zh_en", translation_factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
train = filtered.select_by("clean", "usable")
```

`anydataset.quality.translation.Predicate` labels text pairs as `clean`,
`usable`, `review`, or `reject`. The first built-in profile is WMT19 `zh-en`;
other language pairs should pass an explicit `Profile`.

`anydataset.quality.speech.Predicate` scans audio items with same-role text and
labels samples as `accept` or `reject` based on UTMOS, chrF, duration-per-text
unit, peak amplitude, and optional WER/BLEU thresholds:

```python
from anydataset import AnyDataset, FilterRule, Source, Spec
from anydataset.quality.speech import Predicate as SpeechQuality

def speech_dataset_factory():
    return AnyDataset(
        Spec(source=Source.STORE, path="/data/speech-quality-input", split="train")
    )


def speech_factory():
    return SpeechQuality()

filtered = FilterRule("speech_quality_v1", speech_factory).apply(
    dataset_factory=speech_dataset_factory,
    metrics=True,
)
accepted = filtered.select_by("accept")
```

Speech quality warnings such as missing waveform or missing same-role text are
audit signals in the metrics payload. A non-finite waveform is a hard rejection
before evaluator execution; otherwise a checked audio item is rejected when it
fails a configured threshold.

## Store

`DatasetWriter` writes canonical samples to a self-describing store. The same
store can be read back through `Source.STORE`.

```python
import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.store import DatasetWriter
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
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

Readers accept only store `schema_version: 2`. The preceding canonical store
format used the same sample manifest and directory layout, but had no dataset
schema version and keyed view manifests by `sample_id`. Migrate that format
offline into a new directory; the source is never modified, and the destination
is published only after its manifests, coverage, shards, and payload keys pass
the v2 checks:

```bash
anydataset-store migrate /data/my_anydataset_v1 /data/my_anydataset_v2
```

The equivalent Python API is
`migrate_store("/data/my_anydataset_v1", "/data/my_anydataset_v2")` from
`anydataset.store`.

Older layouts or v1 manifests that do not exactly match that canonical schema
must be re-materialized with `DatasetWriter`; migration does not guess missing
fields or alignment.

`AudioView.FILE` payloads are extracted under
`$ANYDATASET_HOME/cache/store-files`. A reader that selected the file view holds
a shared lease for its lifetime, so cleanup cannot invalidate a returned path
while that reader remains reachable. Hold an explicit lease when the path must
outlive the reader, then clean that physical store when no reader or explicit
lease is active:

```python
from anydataset.store import cleanup_store_files, lease_store_files

with lease_store_files("/data/my_anydataset"):
    retained_path = dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
    del dataset
    consume(retained_path)

cleanup_store_files("/data/my_anydataset")
```

The equivalent maintenance command is
`anydataset-store cleanup-files /data/my_anydataset`. Cleanup raises instead of
deleting leased files, including when the reader is in another process. There
is no automatic eviction; after an explicit cleanup, later access extracts the
payload again.

Views are stored under `{role}/{modality}/{view}/`; payloads live in that
view directory's `shards/` files. `ViewMaterializer` writes derived views to a
delta store. By default it writes only the provider output view; it does not
copy input views or metadata. Open both stores through `Source.STORE`, combine
them with logical `merge()`, and call `write()` only when you need a
self-contained store. Base and delta stores are aligned by `sample_index`;
callers are responsible for materializing views from the same ordered dataset.

```python
from anydataset import AnyDataset, Source, Spec
from anydataset.store import ViewMaterializer
from anydataset.types import AudioView

class ToyLongCat:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, sample_rate = views[AudioView.WAVEFORM]
        return waveform.transpose(0, 1).to(torch.int64)

def dataset_factory():
    return AnyDataset(Spec(source=Source.STORE, path="/data/my_anydataset"))


def provider_factory(device: str):
    return ToyLongCat()


delta = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    input_id="my-audio-v1",
    provider_id="toy-longcat-v1",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="cpu",
)

merged = AnyDataset(Spec(source=Source.STORE, path="/data/my_anydataset")).merge(
    AnyDataset(Spec(source=Source.STORE, path=str(delta)))
)

merged.write("/data/my_anydataset_with_longcat")
```

When a delta must carry selected input fields, declare them with the existing
schema contract instead of copying the whole sample:

```python
from anydataset.types import Modality, Role, TextMeta, TextReq, TextView

keep_schema = {
    (Role.DEFAULT, Modality.TEXT): TextReq(
        views=frozenset({TextView.TEXT}),
        meta=frozenset({TextMeta.LANG}),
    )
}
materializer = ViewMaterializer(
    "/data/my_anydataset_longcat",
    keep_schema=keep_schema,
)
```

`keep_schema` fields must exist in the input. A selected view that conflicts
with the provider output raises instead of overwriting it.

`merge()` returns a map-style logical dataset and never mutates either physical
store. It indexes both sides with the same integer index, like `zip(strict=True)`:
both sides must be map-style datasets with the same length. The right-hand side
may add new items or new views to an existing item; duplicate views fail, and
duplicate metadata keys are allowed only when the values are equal. Runtime
sharding happens on the merged dataset, so both sides share the same global
index. To publish a complete store, call `write()` on the merged dataset.

`write()` can materialize parts in parallel. `num_shards` controls writer
processes, while `num_workers` controls the PyTorch `DataLoader` workers inside
each writer process. For parallel writes, pass a picklable module-level
`dataset_factory` so spawned workers construct their own dataset.

For GPU-backed providers, let `devices` control execution. `devices="auto"`
resolves every visible CUDA device, or CPU when CUDA is unavailable. One
resolved device runs in the calling process; multiple devices use one worker
per device with `Runtime.process_start_method` (`"spawn"` by default), write
worker logs under `$ANYDATASET_HOME/logs/<timestamp>-<pid>/materializer`, and
commit completed fragments when all workers finish. Materializers always use
resumable fragments:
completed provider batches are grouped into checkpoint chunks under a hidden
sibling resume directory, and reruns skip completed global sample indexes
before atomically committing the final store. `commit_samples` controls that
checkpoint granularity and defaults to `max(batch_size, 32)` to avoid excessive
small resume files; lower it when a workload needs finer recovery points.
Resume compatibility includes an automatically derived identity for both
factories. Set `input_id` and `provider_id` to explicit semantic versions when
the input snapshot or provider behavior depends on state that the callables do
not capture, such as mutable files or checkpoint contents. These IDs augment,
rather than replace, the factory identities; changing either one quarantines
the old resume directory instead of reusing incompatible fragments.
Multi-device materialization uses the configured process start method, so
`dataset_factory` and `provider_factory` must be picklable, module-level
callables when that method is `"spawn"`. Like filtering,
multi-device materialization owns its offline worker processes and should not
be launched from inside an existing DDP training process.
Call `write()` from the application main process, not from a PyTorch
`DataLoader` worker or another daemonic process. Multi-device mode creates one
explicitly non-daemonic materializer process per device, and each materializer
process may create `num_workers` DataLoader readers. Switching between `fork`
and `spawn` does not remove Python's restriction on daemonic processes creating
children.
Pass `num_workers` to let each materializer process read samples through a
PyTorch `DataLoader`; this is useful when `parse_fn` does CPU-heavy work such
as file-to-waveform decoding. The materializer sets rank environment variables
for its device workers, and datasets combine rank and DataLoader worker state
inside their runtime shard logic so each sample is covered once.
`write_workers` controls background fragment writer threads inside each
materializer worker; the default is one writer so provider execution can
overlap with store writes. `write_prefetch` bounds pending write jobs.

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

Providers opt into model-side batching by implementing `call_batch(batch)` and
by passing `batch_size` to the materializer. The `batch` argument is the same
`Batch(sample, masks)` object returned by `collate_fn`. `batch_size=1` uses the
per-sample `__call__` path; `batch_size>1` requires `call_batch` and raises a
`TypeError` when it is missing.
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
before writing samples to the store. Each sample stores one integer
`[frame, codebook]` tensor. Collation produces `[batch, frame, codebook]` and a
`[batch, frame]` mask. The dataset layer preserves the complete ordered
codebook axis and does not assign semantic or acoustic meaning to individual
codebooks. `CodecProvider` validates every output column against the codec
contract when it generates a view: each id in column `k` must satisfy
`0 <= id < codebook_sizes[k]`. Store manifests do not carry `codebook_sizes`,
so directly loaded store views are not range-checked by readers or collation.

`ModalityMaterializer` adds a missing modality under the same role. The
provider declares its output view; the materializer infers the output modality
from that view and uses the role's single remaining modality as input. It raises
when the output modality already exists or when the input modality is ambiguous.
Generated items start with empty metadata.

```python
from anydataset.store import ModalityMaterializer
from anydataset.types import AudioView, TextView


class ToyTTS:
    output = AudioView.WAVEFORM

    def __call__(self, views):
        text = views[TextView.TEXT]
        return synthesize(text)


def tts_provider_factory(_device: str):
    return ToyTTS()


delta = ModalityMaterializer(
    output_dir="/data/my_anydataset_tts",
).write(
    dataset_factory=dataset_factory,
    provider_factory=tts_provider_factory,
    devices="cpu",
)
```

Built-in providers follow the model/backend name, for example
`MossTTSProvider` for text-to-audio and `WhisperASRProvider` for
audio-to-text. A provider may set `reference_role` when generation also needs an
already-present output modality from that role, such as reference audio for
TTS. The reference role is skipped as an output target and its views are added
to each other role's single input modality.

## Development

```bash
python -m compileall -q src tests examples
python -m pytest -q
```

Additional design notes live in `docs/design.md`, filter cache details in
`docs/filter_cache.md`, and quality-filter notes in
`docs/translation_quality.md` and `docs/speech_quality.md`. Advanced process
ownership and remote model serving are covered in
`docs/provider_service.md`.

## Release

```bash
python scripts/check_release.py
```

The package exposes `anydataset.__version__`, and the release check verifies
that it matches the `pyproject.toml` version before building. The release check
cleans old build artifacts, runs pytest, builds sdist/wheel, runs `twine check`,
and smoke-installs the wheel in an isolated virtual environment. Use
`--skip-build` when only the version and test gate should run.
