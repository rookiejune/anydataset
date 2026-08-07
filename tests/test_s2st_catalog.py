from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
import torch

from anydataset.dataset import MapStyleABC
from anydataset.filter import FilterRule
from anydataset.synthesis.s2st import (
    CatalogPublisher,
    PairIndexRecord,
    S2STDataset,
    S2STLayout,
    S2STStage,
    S2STView,
    SnapshotManifest,
    catalog_source_locations,
    status,
    store_digest,
    validate_upstream,
)
from anydataset.types import (
    AudioItem,
    AudioView,
    Lang,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)


def test_catalog_serializes_successive_concurrent_publications(tmp_path: Path) -> None:
    first_manifest, first_records = _snapshot(
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    second_manifest, second_records = _snapshot(
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )
    entered = Event()
    release = Event()
    first_publisher = _BlockingPublisher(
        tmp_path,
        "lineage",
        entered=entered,
        release=release,
    )
    second_publisher = CatalogPublisher(tmp_path, "lineage")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_publisher.publish,
            first_manifest,
            first_records,
        )
        assert entered.wait(timeout=5)
        second = executor.submit(
            second_publisher.publish,
            second_manifest,
            second_records,
        )
        try:
            with pytest.raises(FutureTimeoutError):
                second.result(timeout=0.05)
        finally:
            release.set()
        first_catalog = first.result(timeout=5)
        second_catalog = second.result(timeout=5)

    assert first_catalog.latest_snapshot_id == "tts-0"
    assert second_catalog.latest_snapshot_id == "tts-1"
    assert second_catalog.sample_count == 2


def test_publish_retry_requires_identical_manifest_data(tmp_path: Path) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    manifest, records = _snapshot(
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=(
            (0, Lang.EN, Lang.ZH, True),
            (0, Lang.EN, Lang.FR, False),
        ),
        total_sources=1,
    )

    published = publisher.publish(manifest, records)

    assert publisher.publish(manifest, records) == published
    for change in (
        {"added_sources": 0},
        {"total_sources": 2},
        {"coverage": (("en", 1), ("zh", 1))},
    ):
        with pytest.raises(ValueError, match="different catalog data"):
            publisher.publish(replace(manifest, **change), records)


def test_stage_catalogs_publish_and_load_independently(tmp_path: Path) -> None:
    source_publisher = CatalogPublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.SOURCE,
    )
    translation_publisher = CatalogPublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    source_manifest, records = _snapshot(
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
        stage=S2STStage.SOURCE,
    )

    source_publisher.publish(source_manifest, records)

    source_status = status(tmp_path, "lineage", stage=S2STStage.SOURCE)
    assert source_status.stage is S2STStage.SOURCE
    assert source_status.snapshot_count == 1
    assert status(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    ).missing
    assert status(tmp_path, "lineage").missing
    with S2STDataset(
        tmp_path,
        "lineage",
        stage=S2STStage.SOURCE,
    ) as dataset:
        assert dataset.stage is S2STStage.SOURCE
        assert len(dataset) == 1
    with pytest.raises(FileNotFoundError):
        S2STDataset(
            tmp_path,
            "lineage",
            stage=S2STStage.TRANSLATION,
        )

    translation_manifest, records = _snapshot(
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
        stage=S2STStage.TRANSLATION,
    )
    translation_publisher.publish(translation_manifest, records)

    with S2STDataset(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    ) as dataset:
        assert dataset.stage is S2STStage.TRANSLATION
        assert len(dataset) == 1
    assert (tmp_path / "catalogs/source.json").is_file()
    assert (tmp_path / "catalogs/translation.json").is_file()
    assert not (tmp_path / "catalog.json").exists()


def test_dataset_refresh_absorbs_append_only_catalog_growth(
    tmp_path: Path,
) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    first = S2STDataset(tmp_path, "lineage")

    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )

    first.refresh()

    assert first.snapshot_id == "tts-1"
    assert first.snapshot_count == 2
    assert len(first) == 2
    assert [_text(sample, Role.SOURCE) for sample in first.__getitems__((0, 0, 1))] == [
        "source-0",
        "source-0",
        "source-1",
    ]
    first.close()


def test_s2st_dataloader_refreshes_catalog_each_cycle(
    tmp_path: Path,
) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    dataset = S2STDataset(tmp_path, "lineage")
    loader = dataset.dataloader(
        costs=lambda _row: 1,
        max_batch_memory=1,
        max_batch_samples=1,
        materialize_callable_costs=True,
        shuffle=False,
        collate_fn=list,
    )

    assert len(list(loader)) == 1
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )

    loader.set_epoch(1)
    batches = list(loader)

    assert len(batches) == 2
    assert len(dataset) == 2
    assert dataset.snapshot_id == "tts-1"
    assert dataset.snapshot_count == 2
    dataset.close()


def test_s2st_worker_copy_refreshes_when_planned_index_exceeds_its_view(
    tmp_path: Path,
) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    dataset = S2STDataset(tmp_path, "lineage")
    worker_copy = S2STDataset(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )

    dataset.refresh()

    assert len(dataset) == 2
    assert len(worker_copy) == 1
    assert _text(worker_copy[1], Role.SOURCE) == "source-1"
    assert len(worker_copy) == 2
    assert worker_copy.snapshot_id == "tts-1"
    worker_copy.close()
    dataset.close()


def test_s2st_refresh_failure_keeps_previous_dataset_usable(tmp_path: Path) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    dataset = S2STDataset(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )
    entry = publisher.load().entries[-1]
    manifest = tmp_path / entry.store_path / "dataset.json"
    contents = manifest.read_bytes()
    manifest.write_bytes(contents[:-1] + b" ")

    with pytest.raises(ValueError, match="store identity changed"):
        dataset.refresh()

    assert dataset.snapshot_id == "tts-0"
    assert len(dataset) == 1
    assert _text(dataset[0], Role.SOURCE) == "source-0"
    dataset.close()


def test_s2st_decisions_follow_dataset_owned_snapshot_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    decisions = FilterRule(
        "s2st-accept",
        lambda: lambda _sample: "accept",
    ).bind(dataset_factory=lambda: S2STDataset(tmp_path, "lineage"))

    first = decisions.produce(device="cpu", write_workers=0)

    assert first.expected_snapshots == 1
    assert first.completed_snapshots == 1
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )

    grown = decisions.status()
    assert grown.expected_snapshots == 2
    assert grown.completed_snapshots == 1

    complete = decisions.produce(device="cpu", write_workers=0)

    assert complete.complete
    assert complete.completed_snapshots == 2
    with decisions.load() as dataset:
        assert len(dataset) == 2


def test_dataset_status_and_missing_initial_snapshot(tmp_path: Path) -> None:
    missing = status(tmp_path, "lineage")

    assert missing.missing
    assert missing.snapshot_count == 0
    assert missing.sample_count == 0
    assert not missing.sealed
    with pytest.raises(FileNotFoundError):
        S2STDataset(tmp_path, "lineage")


def test_dataset_detects_historical_store_mutation(tmp_path: Path) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    entry = publisher.load().entries[0]
    manifest = tmp_path / entry.store_path / "dataset.json"
    contents = manifest.read_bytes()
    manifest.write_bytes(contents[:-1] + b" ")

    with pytest.raises(ValueError, match="store identity changed"):
        S2STDataset(tmp_path, "lineage")
    with pytest.raises(ValueError, match="store identity changed"):
        status(tmp_path, "lineage")


def test_sources_view_does_not_repeat_language_backfill(tmp_path: Path) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((0, Lang.EN, Lang.FR, False),),
        total_sources=1,
    )
    publisher.seal()

    dataset = S2STDataset(
        tmp_path,
        "lineage",
        view=S2STView(layout=S2STLayout.SOURCES),
    )

    assert len(dataset) == 1
    assert set(dataset[0]) == {
        (Role.SOURCE, Modality.TEXT),
        (Role.SOURCE, Modality.AUDIO),
    }
    assert dataset.sealed
    dataset.close()


def test_catalog_summary_locates_first_source_rows(tmp_path: Path) -> None:
    publisher = CatalogPublisher(tmp_path, "lineage")
    _publish(
        publisher,
        tmp_path,
        revision=0,
        previous=None,
        start=0,
        pairs=((0, Lang.EN, Lang.ZH, True),),
        total_sources=1,
    )
    _publish(
        publisher,
        tmp_path,
        revision=1,
        previous="tts-0",
        start=1,
        pairs=((1, Lang.ZH, Lang.EN, True),),
        total_sources=2,
    )

    locations = catalog_source_locations(
        publisher.load(),
        {("slot", 0), ("slot", 1), ("missing", 0)},
    )

    assert set(locations) == {("slot", 0), ("slot", 1)}
    assert locations[("slot", 0)][0].snapshot_id == "tts-0"
    assert locations[("slot", 1)][0].snapshot_id == "tts-1"


def test_exact_stage_parent_is_required() -> None:
    source = SnapshotManifest(
        lineage_id="lineage",
        config_revision="config",
        revision=0,
        stage=S2STStage.SOURCE,
        snapshot_id="source-0",
        upstream_snapshot_id=None,
        upstream_digest=None,
        previous_snapshot_id=None,
        added_sources=1,
        added_pairs=1,
        total_sources=1,
        total_pairs=1,
        coverage=(("all", 1),),
        store_path="source/0",
        store_digest="source-digest",
    )
    translation = SnapshotManifest(
        lineage_id="lineage",
        config_revision="config",
        revision=0,
        stage=S2STStage.TRANSLATION,
        snapshot_id="translation-0",
        upstream_snapshot_id="source-0",
        upstream_digest="source-digest",
        previous_snapshot_id=None,
        added_sources=1,
        added_pairs=1,
        total_sources=1,
        total_pairs=1,
        coverage=(("all", 1),),
        store_path="translation/0",
        store_digest="translation-digest",
    )

    validate_upstream(translation, source)
    with pytest.raises(ValueError, match="digest"):
        validate_upstream(replace(translation, upstream_digest="wrong"), source)


class _PairDataset(MapStyleABC):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sequence, source, target, _first = self.rows[index]
        return {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"source-{sequence}"},
                meta={TextMeta.LANG: source},
            ),
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.zeros(1, 4), 16000)}
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"target-{sequence}-{target.value}"},
                meta={TextMeta.LANG: target},
            ),
            (Role.TARGET, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.ones(1, 4), 16000)}
            ),
        }


class _BlockingPublisher(CatalogPublisher):
    def __init__(self, *args, entered, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.release = release

    def _publish_locked(self, manifest, records):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("timed out waiting to release test publication")
        return super()._publish_locked(manifest, records)


def _publish(
    publisher,
    root,
    *,
    revision,
    previous,
    start,
    pairs,
    total_sources=None,
):
    manifest, records = _snapshot(
        root,
        revision=revision,
        previous=previous,
        start=start,
        pairs=pairs,
        total_sources=total_sources,
    )
    publisher.publish(manifest, records)
    return manifest, records


def _snapshot(
    root,
    *,
    revision,
    previous,
    start,
    pairs,
    total_sources=None,
    stage=S2STStage.TTS,
):
    store = root / "stores" / f"{stage.value}-{revision}"
    _PairDataset(pairs).write(store)
    records = tuple(
        PairIndexRecord(
            pair_id=f"slot:{sequence}->{target.value}",
            source_slot="slot",
            source_row=sequence,
            source_sequence=sequence,
            source_language=source.value,
            target_language=target.value,
            speaker_id="Vivian",
            first_for_source=first,
        )
        for sequence, source, target, first in pairs
    )
    total_sources = len(pairs) if total_sources is None else total_sources
    manifest = SnapshotManifest(
        lineage_id="lineage",
        config_revision="config",
        revision=revision,
        stage=stage,
        snapshot_id=f"{stage.value}-{revision}",
        upstream_snapshot_id=(
            None
            if stage is S2STStage.SOURCE
            else (
                f"source-{revision}"
                if stage is S2STStage.TRANSLATION
                else f"translation-{revision}"
            )
        ),
        upstream_digest=(
            None
            if stage is S2STStage.SOURCE
            else (
                f"source-digest-{revision}"
                if stage is S2STStage.TRANSLATION
                else f"translation-digest-{revision}"
            )
        ),
        previous_snapshot_id=previous,
        added_sources=max(0, total_sources - start),
        added_pairs=len(pairs),
        total_sources=total_sources,
        total_pairs=start + len(pairs),
        coverage=(("all", start + len(pairs)),),
        store_path=str(store.relative_to(root)),
        store_digest=store_digest(store),
    )
    return manifest, records


def _text(sample, role):
    return sample[(role, Modality.TEXT)].views[TextView.TEXT]
