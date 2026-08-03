# anydataset

[简体中文](README.zh-CN.md)

`anydataset` is a small PyTorch dataset layer for mapping different physical
sources into one canonical `Sample` shape:

```python
Sample = Mapping[tuple[Role, Modality], AudioItem | ImageItem | TextItem]
```

The source layer only prepares and iterates raw rows. Presets decide how those
rows are parsed into canonical samples.

Public import paths and private implementation boundaries are documented in
[Public API Boundaries](docs/public_api.md).

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

Model-backed providers are optional. Install the
matching extra before using them: `anytrain[longcat]` for `LongCatProvider`,
`anytrain[speech]` for `WhisperASRProvider` and the default speech-quality
evaluator, `anydataset[text]` for `TextAcceptability` and `ChineseGEC`, or
`anytrain[moss-tts]` plus `anydataset[audio]` for `MossTTSProvider`.

## Presets

Use `AnyDataset.preset()` when a built-in dataset already knows both its source
and parser. Built-in presets are all map-style: `MNIST`, `CIFAR10`, `FSD50K`,
`COMMON_VOICE`, `FLEURS`, `LIBRISPEECH_ASR`, `ESC50`, `NSYNTH`, and `WMT19`.
Built-in presets are not exposed on `IterableAnyDataset`; use it only for custom
sources that are truly stream-only. Both constructors accept `transforms=...`
for item-level transforms after parsing.

```python
from anydataset import AnyDataset
from anydataset.types import AudioView, Modality, Role

dataset = AnyDataset.preset("fleurs", split="validation")
sample = dataset[0]

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
downloads, and the source cache identity. Internally it uses the generic
`hf-files` physical source; the FSD50K-specific mapping lives in the preset.

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
files = resolve_dataset("hf-files://org/files:train")
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

## Cost-aware batches

Map-style `AnyDataset` can plan dynamic batches from lightweight index-level
costs without parsing the full training sample. Pass `None` for unit-cost
fixed-size batches, a stable integer iterable aligned with global dataset
indexes, or a callable that maps the lightweight row used by `parse_fn` to an
integer cost. A scalar integer is not accepted because constant sample cost
carries no extra information; use `None` for unit-cost batches. Iterable costs
must have the same length as the dataset;
values are read lazily before the corresponding samples are materialized
through `parse_fn`.

```python
from anydataset import AnyDataset


dataset = AnyDataset(spec, parse_fn=parse)
loader = dataset.dataloader(
    costs=lengths,
    max_batch_memory=64_000,
    planning_window=256,
    distributed_plan_window=32,
    shuffle=True,
    collate_fn=collate,
    num_workers=4,
)
```

For store-backed datasets, callable costs receive a manifest row rather than a
materialized payload sample, so length metadata can be read without loading
audio tensors.

The planner treats batch memory and distributed compute as the sum of selected
sample costs. Each sample cost must be a positive integer. Within each planning
window, it greedily adds the fitting sample that makes the batch as full as
possible without exceeding `max_batch_memory`; `max_batch_samples` can cap the
number of samples per planned batch. Planning keeps only a bounded lookahead;
stopping an epoch early does not read costs for the unseen tail. A complete
epoch necessarily reads every selected sample cost once, so expensive lengths
should be persisted in row metadata instead of derived by materializing samples.
With no custom sampler, the dataset builds the rank-local read plan behind the
single `shuffle` flag, and distributed planning never reassigns a planned batch
to a different rank. `StoreDataset` overrides that private plan to shuffle
payload shard groups first and then shuffle sample indexes inside each group.
In DDP, `distributed_plan_window` bounds how many rank-local batch plans are
generated before synchronizing step counts; lower values reduce first-batch
latency when cost lookup or packing is expensive.
Set `ANYDATASET_DEBUG_DDP_PLANS=1` to log each DDP planning chunk before and
after synchronization.
Every group is sliced across ranks, so a store with one payload shard still
feeds every rank while planned batches remain shard-local. DDP synchronizes
plan counts with tensor collectives over bounded windows and only trims
rank-local final batches so all ranks take the same number of steps. Call
`loader.set_epoch(epoch)` before each
distributed epoch to advance the shuffle. The loader also exposes this through
PyTorch's `batch_sampler.sampler.set_epoch(epoch)` contract so trainer frameworks
can advance dataset-owned ordering automatically.

For local JSON, image, or audio files, use `Source.HF` with Hugging Face
`load_dataset(...)` options such as `data_files` or `data_dir`. For raw file
trees hosted on the Hugging Face Hub, use `Source.HF_FILES`; it yields physical
file rows and leaves task-specific parsing to presets or `parse_fn`. For
structured local datasets with canonical samples, use `Source.STORE`.

Built-in enum sources are `Source.HF`, `Source.HF_DISK`,
`Source.HF_FILES`, and `Source.STORE`.
The registry also includes string source keys `tsv` and `sharded_csv`; because
they are registered, they can be used in `Spec(source=...)` and in
`resolve_dataset("<source>://...")` shorthands. `hf-files` lists files under
an optional `path_prefix` or split-aware `path_template`, filters by optional
`suffixes`, downloads individual files on demand, and returns rows with
`repo_id`, `repo_type`, `revision`, `path`, and `local_path`. `tsv` reads a
file path, `<path>/<split>.tsv`, or the same
split under ordered `subdirs` load options, and prepares Parquet parts for
map-style random access (shared with `sharded_csv`); `root_field` is injected
at read time.
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

Pass `operational_load_options=(...)` to `register_source` for source-specific
options that do not change prepared physical data and should therefore be
excluded from `Spec.id`; `prepare_workers` is already treated as operational
for all sources. Register a custom source before constructing any `Spec` with
that source key so its identity policy is fixed before use.

Iterable sources that can select rows without scanning the full stream may also
implement `iter_shard(dataset, *, num_shards, shard_id)`. This source
method must yield `(sample_index, row)` tuples for the exact dense global modulo
shard: indexes start at `shard_id` and advance by `num_shards`. Anydataset
validates tuple shape and index progression before filter or materializer code
sees the rows; the source remains responsible for complete coverage and the
row-to-index association. A raw dataset `shard()` or `iter_shard()`
method alone is not sufficient, because a locally enumerated native shard does
not preserve global indexes.

The built-in `hf-disk`, `hf-files`, `store`, `tsv`, and `sharded_csv`
sources provide this indexed path through random access. Hugging Face
`streaming=True` is rejected; use non-streaming `Source.HF`, `Source.HF_DISK`,
or `Source.HF_FILES`. Dataset-level `iter_shard` is index-preserving and
yields `(sample_index, sample)`. Raw dataset `shard()` methods are never
called opportunistically.

Caches are rooted at `ANYDATASET_HOME`, or `~/.cache/anydataset` when the
environment variable is unset. Source prepare caches live under
`$ANYDATASET_HOME/cache/sources/<spec_id>`, and filter partitions live under
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`.
Runtime warnings and worker logs live under
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`.

Every physical `Spec` field participates in `Spec.id`: `source`, `path`,
`split`, `version`, and physical `load_options`. Operational options that do not
change source content are excluded from `Spec.id` / prepare cache identity. This
includes the global `prepare_workers` option and source-declared options such as
TSV `root_field`. Change `version` or a physical load option when the same path
denotes a different physical snapshot; source prepare caches are reused only for
the resulting identity.

The built-in `tsv` and `sharded_csv` sources keep delimited text files as the
readable source of truth and prepare one Parquet cache part per source file under
`$ANYDATASET_HOME/cache/sources`. Preparation converts changed files in a
spawned process pool and atomically commits the cache manifest. Dataset reads
then use Parquet row groups for map-style random access. Dynamic-batch shuffle
orders those row groups first and only materializes one row group's indexes at a
time, so it does not allocate a full-dataset Python index list. Set
`Spec(load_options={"prepare_workers": 0})` or `1` to disable process-pool
preparation in restricted environments; the default uses bounded automatic
parallelism.

## DataLoader Schemas

`Schema` maps each `(Role, Modality)` reference to the views and metadata that
a training batch needs. `collate_fn(schema)` selects those fields and returns a
`Batch`; it does not fill in missing fields implicitly.

Every dataset exposes `iter_shard(num_shards, shard_id)` for distributed reads;
it yields `(sample_index, sample)` pairs.

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

Keep reusable training inputs as explicit schemas. A project can define a small
helper for common layouts, then pass it to the generic collator:

```python
from anydataset.dataset import collate_fn
from anydataset.types import AudioReq, AudioView, Modality, Role

audio_codec_schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
    )
}
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_fn(audio_codec_schema),
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
labels are stored as their own partitions. `name` is the human-readable rule
name and remains part of the cache identity; optional `rule_id` and `version`
fields add explicit predicate, parser, and transform identity. Changing any of
these fields selects a different cache. When omitted, `rule_id` defaults to
`name` and the legacy name-only cache path remains compatible.

`FilteredDataset(...)` checks whether the rule identity already has a ready cache
for the base dataset. If not, it builds the cache. It selects every available
label by default. Use `select_by(...)` to derive a label view over the same
cache. `FilterRule.apply(...)` is a convenience wrapper that forwards its
`name` and `factory` to `FilteredDataset`.

Use `FilterRule.apply_with_report(...)` when a caller needs wall-clock
observability for a specific apply call without storing run state on the
dataset object:

```python
applied = rule.apply_with_report(dataset_factory=dataset_factory, device="cpu")
filtered = applied.dataset
report = applied.report

if not report.cache_hit and report.logs_dir is not None:
    print(f"filter logs: {report.logs_dir}")
print(
    "filter apply took "
    f"{report.elapsed_seconds:.2f}s "
    f"({report.samples_per_second:.1f} samples/s)"
)
```

`FilterApplyReport` separates dataset construction, cache lookup/build, and
partition-read timings. On hot-cache hits, `report.logs_dir` is `None` and
`report.cache_build_seconds` is `0.0`; the report measures apply-call
overhead, not cache schema metadata.

Filter cache identity is automatic for physical datasets and filtered views.
For a mutable or application-owned input, pass a non-empty `input_id` to
`apply()` or `FilteredDataset(...)`. The ID versions the entire input snapshot
and augments the automatic class, `Spec`, and sample-count identity. Change it
when input content or ordering changes; `FilterRule.rule_id` and `version` identify
predicate semantics. A store's manifest provenance is included automatically,
so materializer `input_id` and `provider_id` changes also produce a new filter
cache identity. The explicit ID is preserved by the filtered
`dataset_factory`, pickle, and chained filters.

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
rather than from inside an existing DDP training process. The dataset entry
point is always `dataset_factory=...`. With the default `"spawn"` process start
method, both dataset and predicate factories should be module-level picklable
callables.
Pass `num_workers` to let each execution process read samples through a PyTorch
`DataLoader`; `batch_size` controls the loader batch size.
Predicates may implement `call_batch(samples)` to consume that batch directly.
The method must return an ordered sequence with exactly one output per input;
predicates without it continue to run through per-sample `__call__`.

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

For rare CPU-only hard failures after an accept partition is selected, wrap the
map-style dataset with `RejectReplaceDataset` (sequential look-ahead, then a
worker-local accept buffer). It is not a substitute for cached partitions; see
[`docs/online_filter.md`](docs/online_filter.md).

## Quality Rules

Quality modules provide reusable rule classes for `FilterRule`; they do not own
dataset loading or cache naming.

`FilterRule` accepts map-style inputs. WMT19 is map-style, so filter it directly:

```python
from anydataset import (
    AnyDataset,
    FilterRule,
    Lang,
    Preset,
)
from anydataset.quality.rules import QualityChain, Rule
from anydataset.quality.text import TextQuality
from anydataset.quality.translation import TranslationQuality
from anydataset.types import Role

source = AnyDataset.preset(
    "wmt19", source_lang="zh", target_lang="en"
)


def dataset_factory():
    return AnyDataset.preset("wmt19", source_lang="zh", target_lang="en")


def translation_factory():
    return QualityChain(
        (
            Rule(
                "source_text",
                TextQuality(role=Role.SOURCE, lang=Lang.ZH),
            ),
            Rule(
                "target_text",
                TextQuality(role=Role.TARGET, lang=Lang.EN),
            ),
            Rule(
                "pair",
                TranslationQuality.from_preset(
                    Preset.WMT19,
                    source_lang=Lang.ZH,
                    target_lang=Lang.EN,
                ),
            ),
        )
    )

filtered = FilterRule("mt_quality_rules_v1_zh_en", translation_factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
train = filtered.select_by("accept")
```

`anydataset.quality.text.TextQuality` checks each text item independently.
`anydataset.quality.translation.TranslationQuality` checks only source/target
pair consistency. Compose them with `anydataset.quality.rules.QualityChain` to
control the filtering order and keep per-rule metrics. Canonical language meta
uses `anydataset.Lang`; map external dataset labels at the boundary with
`anydataset.remap_lang(...)`.

`anydataset.quality.speech.SpeechQuality` scans audio items with same-role text and
labels samples as `accept` or `reject` based on UTMOS, chrF, duration-per-text
unit, peak amplitude, and optional WER/BLEU thresholds:

```python
from anydataset import AnyDataset, FilterRule, Source, Spec
from anydataset.quality.speech import SpeechQuality

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

Pass an existing `CodecProvider` as `SpeechQuality(codec_provider=provider)` to
evaluate reconstructed audio for `provider.output`. This path reads only that
codec view, batches equal-length codes through `provider.codec.decode()`, and
uses `provider.codec.sample_rate`; it never falls back to an original waveform.
Version the filter rule when the codec view, checkpoint, or decoder
configuration changes.

## Store

`DatasetWriter` writes canonical samples to a self-describing store. The same
store can be read back through `Source.STORE`.

```python
import torch

from anydataset import AnyDataset
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

dataset = AnyDataset.from_store(
    "/data/my_anydataset",
    views=((Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),),
)
restored = dataset[0]
```

`dataset_id` is optional. When omitted, it defaults to the final component of
the expanded `output_dir`, or `"dataset"` when that path has no name.
`provenance` records semantic input state in the dataset manifest. It accepts
only non-empty string `input_id` and `provider_id` values; use them to version
input content or provider behavior that the dataset path and configuration do
not capture. Provenance participates in downstream filter cache identity.

`write(samples)` and `write(dataset_factory=...)` are mutually exclusive. The
defaults (`num_shards=1`, `num_workers=0`) write in the calling process and
accept either form. If `num_shards > 1` or `num_workers > 0`, a
`dataset_factory` is required so every process constructs its own dataset:

```python
from anydataset import AnyDataset
from anydataset.store import DatasetWriter


def dataset_factory():
    return AnyDataset.from_store("/data/source_anydataset")


if __name__ == "__main__":
    DatasetWriter(
        "/data/parallel_copy",
        provenance={"input_id": "source-2026-07-30"},
        num_shards=4,
        num_workers=2,
        prefetch_factor=4,
    ).write(dataset_factory=dataset_factory)
```

`num_shards` is the number of store writer processes. `num_workers` is the
number of PyTorch `DataLoader` readers inside each writer process, so a fully
parallel run can create `num_shards * num_workers` loader workers in addition
to the writers. `prefetch_factor` is the number of one-sample loader batches
prefetched by each loader worker; it is used only when `num_workers > 0` and
defaults to `2` when omitted. Parallel factories must be picklable (normally a
module-level callable under the default `spawn` start method), and the returned
dataset must be map-style or implement `iter_shard()`. Run the writer
from the application main process. When loader workers are enabled and the
dataset exposes `prepare()`, the writer calls it once in the parent before
spawning so shared source metadata can be prepared safely.

`AnyDataset.from_store(..., views=...)` loads only the selected view manifests
and payloads. The selection is preserved across pickle/spawn and participates
in filter cache identity, while the physical `Spec` continues to identify the
store itself.

Store payloads are written to tar shards per view. The same
`dataset.dataloader(..., shuffle=True)` entry point remains the only shuffle
control for store training: when the prepared dataset is a `StoreDataset`, the
loader first shuffles payload shard groups, slices every group across ranks,
then shuffles rank-local indexes within each group, and plans batches without
crossing shard-group boundaries. Use
`seed=...` and `loader.set_epoch(epoch)` for reproducible epoch changes.

Store readers default to the current `schema_version: 3`. Version 2 stores have
no provenance and are treated as legacy: `read_store_dataset(...)`,
`AnyDataset.from_store(...)`, and `Source.STORE` reject them unless the caller
explicitly chooses `legacy_policy="allow"` or `legacy_policy="warn"`. New v3
stores persist materializer `input_id` and `provider_id`.

> Warning: v2 compatibility is legacy read support, not a recommended production
> format. `legacy_policy="warn"` emits a `RuntimeWarning`; `"allow"` is a
> silent explicit opt-in. In both cases, missing provenance is treated as empty,
> so downstream cache identity cannot distinguish input or provider semantic
> versions. Rematerialize or migrate to v3 before publishing a store or using it
> as the basis for cache-sensitive derived data.

The preceding canonical store format used the same sample manifest and directory
layout, but had no dataset schema version and keyed view manifests by
`sample_id`. Migrate that format offline into a new directory; the source is
never modified, and the destination is published only after its manifests,
coverage, shards, and payload keys pass the v3 checks:

```bash
anydataset-store migrate /data/my_anydataset_v1 /data/my_anydataset_v3
```

The equivalent Python API is
`migrate_store("/data/my_anydataset_v1", "/data/my_anydataset_v3")` from
`anydataset.store`.

Older layouts or v1 manifests that do not exactly match that canonical schema
must be re-materialized with `DatasetWriter`; migration does not guess missing
fields or alignment.

Use the integrity maintenance helper before publishing or moving a store. `fast` checks
manifest references and shard existence, `normal` also parses every referenced
tar archive and rejects invalid or duplicate file members, and the default
`full` level additionally reads referenced payload bodies and rejects extra
payload members that are not present in the manifest:

```python
from pathlib import Path
from anydataset.store import validate_store_payloads

validate_store_payloads((Path("/data/my_anydataset"),), level="full")
```

Store readers retain manifest and tar handles for repeated access. Close them
deterministically with `dataset.close()` or use `read_store_dataset(...)` as a
context manager when the reader lifetime is bounded.

Store payloads are read with PyTorch's safe weights-only deserialization by
default. Tensor payloads, strings, and ordinary Python containers remain
supported. Stores containing custom Python payload objects require an explicit
`AnyDataset.from_store(..., unsafe_pickle_payloads=True)` or
`read_store_dataset(..., unsafe_pickle_payloads=True)` opt-in, and should only be
read from trusted sources.

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
view directory's `shards/` files. `ViewMaterializer` always publishes a
standalone store. By default it writes only the provider output view; input
views and metadata are retained only when selected with `keep_schema`.

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


output = ViewMaterializer(
    output_dir="/data/my_anydataset_longcat",
    input_id="my-audio-v1",
    provider_id="toy-longcat-v1",
).write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="cpu",
)

dataset = AnyDataset(Spec(source=Source.STORE, path=str(output)))
```

When the output must carry selected input fields, declare them with the existing
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
with the provider output raises instead of overwriting it. To publish multiple
derived views, run another materializer against the previous standalone store
and explicitly retain the fields needed by the next stage.

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
checkpoint granularity and defaults to `max(batch_size, 1024)` to avoid
excessive small resume files; lower it when a workload needs finer recovery
points.
Resume compatibility includes an automatically derived identity for both
factories. Set `input_id` and `provider_id` to explicit semantic versions when
the input snapshot or provider behavior depends on state that the callables do
not capture, such as mutable files or checkpoint contents. These IDs augment,
rather than replace, the factory identities; changing either one quarantines
the old resume directory instead of reusing incompatible fragments. The same
IDs are written to the final store manifest provenance and participate in
downstream filter cache identity.

Materialization can be intentionally staged without changing the input
identity. Pass `max_new_samples` or explicit increasing `sample_indexes` and
`finalize=False`; the call returns a `MaterializationStatus` and leaves its
hidden fragments available for the next call. The later call must use the same
dataset/provider identities and can omit the bound to process all remaining
indexes and atomically publish the final store:

```python
status = materializer.write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
    max_new_samples=50_000,
    finalize=False,
)
assert status.completed <= 50_000

output = materializer.write(
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
    devices="auto",
)
```

`max_new_samples` and `sample_indexes` are supported for map-style datasets
only. A partial run is not a readable store until finalized. To publish a dense
completed prefix for inspection while keeping the run open, call
`snapshot()`; snapshots do not clean up or advance the resume state:

```python
preview = materializer.snapshot(
    "/data/my_anydataset_audio-50k",
    dataset_factory=dataset_factory,
    provider_factory=provider_factory,
)
```

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
Public defaults are intentionally conservative: they keep single-process,
single-sample execution usable across providers and platforms. Production
workflows should tune `batch_size`, `num_workers`, `prefetch_factor`,
`write_workers`, `write_prefetch`, and `commit_samples` in the calling
script or job wrapper based on the provider, storage backend, and hardware.

```python
def provider_factory(device: str):
    from anydataset.provider.longcat import LongCatProvider

    return LongCatProvider(device=device)


output = ViewMaterializer(
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
If a batch provider raises an out-of-memory error, the materializer clears
available caches and recursively retries the batch as two smaller batches. Each
split writes an immediate progress log with the worker, provider type, failed
batch size, and retry sizes. An out-of-memory error for one sample is not
recoverable and is raised to the caller.

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
Generated items start with empty metadata. Pass `roles={Role.TARGET}` to limit
generation to selected roles; by default every eligible role is materialized.

```python
from anydataset.store import ModalityMaterializer
from anydataset.types import AudioView, Role, TextView


class ToyTTS:
    output = AudioView.WAVEFORM

    def __call__(self, views):
        text = views[TextView.TEXT]
        return synthesize(text)


def tts_provider_factory(_device: str):
    return ToyTTS()


output = ModalityMaterializer(
    output_dir="/data/my_anydataset_tts",
    roles={Role.TARGET},
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
python -m ruff check src tests scripts examples
python -m basedpyright --pythonpath "$(python -c 'import sys; print(sys.executable)')"
python -m pytest -q
```

Additional design notes live in `docs/design.md`, production logging notes in
`docs/logging.md`, filter cache details in `docs/filter_cache.md`, online
reject-replace notes in `docs/online_filter.md`, and quality-filter notes in
`docs/translation_quality.md` and `docs/speech_quality.md`. Advanced process
ownership and remote model serving are covered in `docs/provider_service.md`.

## Release

```bash
python scripts/check_release.py
```

The package exposes `anydataset.__version__`, and the release check verifies
that it matches the `pyproject.toml` version before building. The release check
cleans old build artifacts, runs pytest, builds sdist/wheel, runs `twine check`,
and smoke-installs the wheel in an isolated virtual environment. Use
`--skip-build` when only the version and test gate should run.
