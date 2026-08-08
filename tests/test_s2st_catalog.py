from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from anydataset.dataset import MapStyleABC
from anydataset.filter import FilterRule
from anydataset.synthesis.s2st import (
    PairIndexRecord,
    S2STDataset,
    S2STLayout,
    S2STStage,
    S2STView,
    StageInput,
    StagePublisher,
    status,
)
from anydataset.types import (
    AudioItem,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)


Pair = tuple[int, Lang, Lang, bool]


def test_stages_advance_independently_over_fixed_upstream_watermarks(
    tmp_path: Path,
) -> None:
    source = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.SOURCE,
    )
    translation = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    tts = StagePublisher(tmp_path, "lineage", stage=S2STStage.TTS)

    first_records = _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True), (1, Lang.ZH, Lang.EN, True)),
    )
    translation_input = translation.open_input(max_samples=1)
    assert isinstance(translation_input, StageInput)
    assert (translation_input.start, translation_input.stop) == (0, 1)
    assert len(translation_input) == 1

    _publish_source(
        source,
        tmp_path,
        1,
        ((2, Lang.EN, Lang.ZH, True),),
    )
    _publish_stage(translation, translation_input, tmp_path, 0, first_records[:1])

    translation_status = translation.status()
    assert translation_status.published_samples == 1
    assert translation_status.expected_samples == 3
    assert translation_status.pending_samples == 2

    tts_input = tts.open_input()
    assert isinstance(tts_input, StageInput)
    next_translation = translation.open_input()
    assert isinstance(next_translation, StageInput)
    _publish_stage(
        translation,
        next_translation,
        tmp_path,
        1,
        first_records[1:2],
    )
    _publish_stage(tts, tts_input, tmp_path, 0, first_records[:1])

    assert tts.status().published_samples == 1
    assert tts.status().expected_samples == 2
    assert source.status().published_samples == 3


def test_stage_input_can_expose_part_of_one_internal_publication(
    tmp_path: Path,
) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    translation = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    records = _publish_source(
        source,
        tmp_path,
        0,
        (
            (0, Lang.EN, Lang.ZH, True),
            (1, Lang.ZH, Lang.EN, True),
            (2, Lang.EN, Lang.ZH, True),
        ),
    )

    first = translation.open_input(max_samples=2)
    assert isinstance(first, StageInput)
    assert len(first) == 2
    assert [_text(first[index], Role.SOURCE) for index in range(2)] == [
        "source-0",
        "source-1",
    ]
    _publish_stage(translation, first, tmp_path, 0, records[:2])

    second = translation.open_input(max_samples=2)
    assert isinstance(second, StageInput)
    assert (second.start, second.stop, len(second)) == (2, 3, 1)
    second.close()


def test_same_stage_rejects_stale_pinned_input(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    translation = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    records = _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    first = translation.open_input()
    stale = translation.open_input()
    assert isinstance(first, StageInput)
    assert isinstance(stale, StageInput)

    _publish_stage(translation, first, tmp_path, 0, records)
    _write_store(tmp_path / "stores/translation-stale", records)
    with pytest.raises(RuntimeError, match="stale"):
        translation.publish(stale, tmp_path / "stores/translation-stale")
    stale.close()


def test_downstream_seal_requires_sealed_complete_upstream(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    translation = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    records = _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    stage_input = translation.open_input()
    assert isinstance(stage_input, StageInput)
    _publish_stage(translation, stage_input, tmp_path, 0, records)

    with pytest.raises(ValueError, match="sealed source"):
        translation.seal()
    source.seal()
    sealed = translation.seal()

    assert sealed.sealed
    assert sealed.caught_up


def test_missing_stage_is_a_valid_empty_read_only_prefix(tmp_path: Path) -> None:
    missing = status(tmp_path, "lineage", stage=S2STStage.TRANSLATION)

    assert missing.missing
    assert missing.published_samples == 0
    assert missing.expected_samples == 0
    assert missing.caught_up
    with S2STDataset(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    ) as dataset:
        assert len(dataset) == 0


def test_open_reader_remains_fixed_after_later_publication(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    first = S2STDataset(tmp_path, "lineage", stage=S2STStage.SOURCE)

    _publish_source(
        source,
        tmp_path,
        1,
        ((1, Lang.ZH, Lang.EN, True),),
    )

    assert len(first) == 1
    with pytest.raises(IndexError):
        first[1]
    with S2STDataset(tmp_path, "lineage", stage=S2STStage.SOURCE) as reopened:
        assert len(reopened) == 2
        assert _text(reopened[1], Role.SOURCE) == "source-1"
    first.close()


def test_dataloader_keeps_the_reader_prefix_fixed(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    dataset = S2STDataset(tmp_path, "lineage", stage=S2STStage.SOURCE)
    loader = dataset.dataloader(
        costs=lambda _row: 1,
        max_batch_memory=1,
        max_batch_samples=1,
        materialize_callable_costs=True,
        shuffle=False,
        collate_fn=list,
    )

    assert len(list(loader)) == 1
    _publish_source(
        source,
        tmp_path,
        1,
        ((1, Lang.ZH, Lang.EN, True),),
    )
    loader.set_epoch(1)

    assert len(list(loader)) == 1
    dataset.close()


def test_new_reader_detects_mutation_while_open_reader_remains_usable(
    tmp_path: Path,
) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    dataset = S2STDataset(tmp_path, "lineage", stage=S2STStage.SOURCE)
    catalog = source._publisher.load()
    entry = catalog.entries[0]
    manifest = tmp_path / entry.store_path / "dataset.json"
    contents = manifest.read_bytes()
    manifest.write_bytes(contents[:-1] + b" ")

    with pytest.raises(ValueError, match="store identity changed"):
        S2STDataset(tmp_path, "lineage", stage=S2STStage.SOURCE)

    assert len(dataset) == 1
    assert _text(dataset[0], Role.SOURCE) == "source-0"
    dataset.close()


def test_sources_view_does_not_repeat_language_backfill(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    _publish_source(
        source,
        tmp_path,
        1,
        ((0, Lang.EN, Lang.FR, False),),
    )
    source.seal()

    with S2STDataset(
        tmp_path,
        "lineage",
        stage=S2STStage.SOURCE,
        view=S2STView(layout=S2STLayout.SOURCES),
    ) as dataset:
        assert len(dataset) == 1
        assert dataset.sealed
        assert set(dataset[0]) == {
            (Role.SOURCE, Modality.TEXT),
            (Role.SOURCE, Modality.AUDIO),
        }


def test_decisions_publish_sample_coverage_over_reopened_stage_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    decisions = FilterRule(
        "s2st-accept",
        lambda: lambda _sample: "accept",
    ).bind(
        dataset_factory=lambda: S2STDataset(
            tmp_path,
            "lineage",
            stage=S2STStage.SOURCE,
        ),
        max_new_samples=1,
    )

    first = decisions.produce(device="cpu", write_workers=0)
    assert (first.completed_samples, first.expected_samples) == (1, 1)

    _publish_source(
        source,
        tmp_path,
        1,
        ((1, Lang.ZH, Lang.EN, True),),
    )
    grown = decisions.status()
    assert (grown.completed_samples, grown.expected_samples) == (1, 2)

    complete = decisions.produce(device="cpu", write_workers=0)
    assert complete.complete
    with decisions.load() as dataset:
        assert len(dataset) == 2


def test_default_tts_reader_observes_only_tts_progress(tmp_path: Path) -> None:
    source = StagePublisher(tmp_path, "lineage", stage=S2STStage.SOURCE)
    translation = StagePublisher(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    )
    records = _publish_source(
        source,
        tmp_path,
        0,
        ((0, Lang.EN, Lang.ZH, True),),
    )
    translation_input = translation.open_input()
    assert isinstance(translation_input, StageInput)
    _publish_stage(translation, translation_input, tmp_path, 0, records)

    with S2STDataset(tmp_path, "lineage") as tts_dataset:
        assert len(tts_dataset) == 0
    with S2STDataset(
        tmp_path,
        "lineage",
        stage=S2STStage.TRANSLATION,
    ) as translation_dataset:
        assert len(translation_dataset) == 1


class _PairDataset(MapStyleABC):
    def __init__(self, records: tuple[PairIndexRecord, ...]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        source = Lang(record.source_language)
        target = Lang(record.target_language)
        return cast(
            Sample,
            {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"source-{record.source_sequence}"},
                meta={TextMeta.LANG: source},
            ),
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.zeros(1, 4), 16_000)}
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={
                    TextView.TEXT: f"target-{record.source_sequence}-{target.value}"
                },
                meta={TextMeta.LANG: target},
            ),
            (Role.TARGET, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.ones(1, 4), 16_000)}
            ),
            },
        )


def _publish_source(
    publisher: StagePublisher,
    root: Path,
    revision: int,
    pairs: tuple[Pair, ...],
) -> tuple[PairIndexRecord, ...]:
    records = _records(pairs)
    store = root / "stores" / f"source-{revision}"
    _write_store(store, records)
    publisher.publish_source(
        store,
        records,
        config_revision=f"config-{revision}",
    )
    return records


def _publish_stage(
    publisher: StagePublisher,
    stage_input: StageInput,
    root: Path,
    revision: int,
    records: tuple[PairIndexRecord, ...],
) -> None:
    store = root / "stores" / f"{publisher.stage.value}-{revision}"
    _write_store(store, records)
    publisher.publish(stage_input, store)
    stage_input.close()


def _write_store(path: Path, records: tuple[PairIndexRecord, ...]) -> None:
    _PairDataset(records).write(path)


def _records(pairs: tuple[Pair, ...]) -> tuple[PairIndexRecord, ...]:
    return tuple(
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


def _text(sample, role: Role) -> str:
    item = cast(TextItem, sample[(role, Modality.TEXT)])
    return cast(str, item.views[TextView.TEXT])
