from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
import json
import tempfile
from threading import Event
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import anydataset.synthesis.s2st.catalog as s2st_catalog_module
import anydataset.synthesis.s2st.dataset as s2st_dataset_module
from anydataset.dataset import MapStyleABC
from anydataset.synthesis.s2st import (
    CatalogPublisher,
    LiveS2STDataset,
    PairIndexRecord,
    S2STLayout,
    S2STStage,
    S2STView,
    SnapshotManifest,
    catalog_source_locations,
    store_digest,
    validate_catalog_entry,
    validate_catalog_store,
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


class S2STCatalogTest(unittest.TestCase):
    def test_catalog_serializes_successive_concurrent_publications(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest, first_records = _snapshot(
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            second_manifest, second_records = _snapshot(
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=((1, Lang.ZH, Lang.EN, True),),
                total_sources=2,
            )
            entered = Event()
            release = Event()
            first_publisher = _BlockingPublisher(
                root,
                "lineage",
                entered=entered,
                release=release,
            )
            second_publisher = CatalogPublisher(root, "lineage")
            second_started = Event()

            def publish_second():
                second_started.set()
                return second_publisher.publish(second_manifest, second_records)

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    first_publisher.publish,
                    first_manifest,
                    first_records,
                )
                self.assertTrue(entered.wait(timeout=5))
                second = executor.submit(publish_second)
                self.assertTrue(second_started.wait(timeout=5))
                try:
                    with self.assertRaises(FutureTimeoutError):
                        second.result(timeout=0.05)
                finally:
                    release.set()
                first_catalog = first.result(timeout=5)
                second_catalog = second.result(timeout=5)

            self.assertEqual(first_catalog.latest_snapshot_id, "tts-0")
            self.assertEqual(second_catalog.latest_snapshot_id, "tts-1")
            self.assertEqual(second_catalog.sample_count, 2)

    def test_publish_retry_requires_identical_manifest_counts_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            manifest, records = _snapshot(
                root,
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

            self.assertEqual(publisher.publish(manifest, records), published)
            changes = (
                {"added_sources": 0},
                {"total_sources": 2},
                {"coverage": (("en", 1), ("zh", 1))},
            )
            for change in changes:
                with self.subTest(change=change):
                    with self.assertRaisesRegex(ValueError, "different catalog data"):
                        publisher.publish(replace(manifest, **change), records)

    def test_publish_recovers_identical_orphan_pair_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            manifest, records = _snapshot(
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            index = root / "indexes" / "00000000-tts-0.jsonl"

            with patch(
                "anydataset.synthesis.s2st.catalog.write_catalog",
                side_effect=RuntimeError("injected catalog commit failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    publisher.publish(manifest, records)

            orphan = index.read_bytes()
            self.assertFalse((root / "catalog.json").exists())
            catalog = publisher.publish(manifest, records)

            self.assertEqual(index.read_bytes(), orphan)
            self.assertEqual(catalog.latest_snapshot_id, manifest.snapshot_id)
            self.assertEqual(publisher.publish(manifest, records), catalog)

    def test_refresh_failure_does_not_publish_partial_in_memory_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            live = LiveS2STDataset(root, "lineage", poll_seconds=0.01, status_seconds=1)
            live.refresh()
            before = live.state_dict()
            before_count = live.sample_count
            _publish(
                publisher,
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=((1, Lang.ZH, Lang.EN, True),),
                total_sources=2,
            )
            _publish(
                publisher,
                root,
                revision=2,
                previous="tts-1",
                start=2,
                pairs=((2, Lang.EN, Lang.FR, True),),
                total_sources=3,
            )
            index = root / publisher.load().entries[-1].index_path
            contents = index.read_bytes()
            original_close = s2st_dataset_module._close_dataset
            original_sha256 = s2st_catalog_module._sha256

            def sha256(value):
                if value == contents:
                    return "injected-corrupt-digest"
                return original_sha256(value)

            with (
                patch.object(
                    s2st_catalog_module,
                    "_sha256",
                    side_effect=sha256,
                ),
                patch.object(
                    s2st_dataset_module,
                    "_close_dataset",
                    wraps=original_close,
                ) as close_dataset,
            ):
                with self.assertRaisesRegex(ValueError, "pair index digest mismatch"):
                    live.refresh()

            self.assertEqual(close_dataset.call_count, 1)
            self.assertEqual(live.state_dict(), before)
            self.assertEqual(live.sample_count, before_count)

            update = live.refresh()
            self.assertIsNotNone(update)
            self.assertEqual(live.snapshot_id, "tts-2")
            self.assertEqual(live.sample_count, 3)
            live.close()

    def test_publish_validates_history_from_incremental_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            manifest, records = _snapshot(
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=((1, Lang.ZH, Lang.EN, True),),
                total_sources=2,
            )
            original_digest = s2st_catalog_module.store_digest

            with (
                patch.object(
                    s2st_catalog_module,
                    "read_pair_index",
                    side_effect=AssertionError("historical pair index was reread"),
                ),
                patch.object(
                    s2st_catalog_module,
                    "store_digest",
                    wraps=original_digest,
                ) as digest,
            ):
                catalog = publisher.publish(manifest, records)

            self.assertEqual(catalog.sample_count, 2)
            self.assertEqual(
                [Path(call.args[0]) for call in digest.call_args_list],
                [root / manifest.store_path],
            )

    def test_incremental_summary_rejects_duplicate_historical_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            manifest, records = _snapshot(
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=((0, Lang.EN, Lang.ZH, False),),
                total_sources=1,
            )

            with patch.object(
                s2st_catalog_module,
                "read_pair_index",
                side_effect=AssertionError("historical pair index was reread"),
            ):
                with self.assertRaisesRegex(ValueError, "pair id already exists"):
                    publisher.publish(manifest, records)

    def test_catalog_summary_locates_first_source_rows_without_reading_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            _publish(
                publisher,
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=(
                    (0, Lang.EN, Lang.FR, False),
                    (1, Lang.ZH, Lang.EN, True),
                ),
                total_sources=2,
            )
            catalog = publisher.load()

            with patch.object(
                s2st_catalog_module,
                "read_pair_index",
                side_effect=AssertionError("pair index was read"),
            ):
                locations = catalog_source_locations(
                    catalog,
                    {("slot", 0), ("slot", 1), ("missing", 0)},
                )

            self.assertEqual(set(locations), {("slot", 0), ("slot", 1)})
            self.assertEqual(locations[("slot", 0)][0].snapshot_id, "tts-0")
            self.assertEqual(locations[("slot", 0)][1], 0)
            self.assertEqual(locations[("slot", 1)][0].snapshot_id, "tts-1")
            self.assertEqual(locations[("slot", 1)][1], 1)

    def test_live_access_uses_store_identity_without_full_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            live = LiveS2STDataset(root, "lineage", poll_seconds=0.01, status_seconds=1)
            entry = publisher.load().entries[0]

            with patch.object(
                s2st_catalog_module,
                "store_digest",
                side_effect=AssertionError("waveform store was fully digested"),
            ):
                validate_catalog_entry(root, entry)
                validate_catalog_store(root, entry)
                live.refresh()

            self.assertEqual(live.sample_count, 1)
            live.close()

    def test_store_identity_detects_same_size_historical_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            entry = publisher.load().entries[0]
            manifest = root / entry.store_path / "dataset.json"
            contents = manifest.read_bytes()
            manifest.write_bytes(contents[:-1] + b" ")
            live = LiveS2STDataset(root, "lineage", poll_seconds=0.01, status_seconds=1)

            with self.assertRaisesRegex(ValueError, "store identity changed"):
                live.refresh()

            live.close()

    def test_catalog_summary_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            path = root / "catalog.json"
            catalog = json.loads(path.read_text(encoding="utf-8"))
            catalog["entries"][0]["index_summary"][0]["target_languages"] = ["fr"]
            catalog["entries"][0]["index_summary"][0]["first_target_language"] = "fr"
            path.write_text(json.dumps(catalog), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "index summary digest"):
                publisher.load()

    def test_live_dataset_refreshes_append_only_catalog_without_resetting_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True), (1, Lang.ZH, Lang.EN, True)),
            )
            live = LiveS2STDataset(root, "lineage", poll_seconds=0.01, status_seconds=1)
            iterator = iter(live)

            first = next(iterator)
            second = next(iterator)
            live.acknowledge()
            _publish(
                publisher,
                root,
                revision=1,
                previous="tts-0",
                start=2,
                pairs=((2, Lang.EN, Lang.FR, True),),
                total_sources=3,
            )
            publisher.seal()
            third = next(iterator)

            self.assertEqual(_text(first, Role.SOURCE), "source-0")
            self.assertEqual(_text(second, Role.SOURCE), "source-1")
            self.assertEqual(_text(third, Role.SOURCE), "source-2")
            self.assertEqual(live.cursor, 2)
            live.acknowledge()
            self.assertEqual(live.cursor, 3)
            with self.assertRaises(StopIteration):
                next(iterator)
            live.close()

    def test_sources_view_does_not_repeat_old_family_language_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            _publish(
                publisher,
                root,
                revision=1,
                previous="tts-0",
                start=1,
                pairs=((0, Lang.EN, Lang.FR, False),),
                total_sources=1,
            )
            publisher.seal()
            live = LiveS2STDataset(
                root,
                "lineage",
                view=S2STView(layout=S2STLayout.SOURCES),
                poll_seconds=0.01,
                status_seconds=1,
            )

            samples = list(live)

            self.assertEqual(len(samples), 1)
            self.assertEqual(
                set(samples[0]),
                {(Role.SOURCE, Modality.TEXT), (Role.SOURCE, Modality.AUDIO)},
            )
            live.close()

    def test_exact_stage_parent_is_required(self):
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
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_upstream(
                SnapshotManifest(
                    **{
                        **translation.__dict__,
                        "upstream_digest": "wrong",
                    }
                ),
                source,
            )

        with self.assertRaisesRegex(ValueError, "config revision"):
            validate_upstream(
                SnapshotManifest(
                    **{
                        **translation.__dict__,
                        "config_revision": "changed",
                    }
                ),
                source,
            )

    def test_live_checkpoint_requires_an_existing_snapshot_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = CatalogPublisher(root, "lineage")
            _publish(
                publisher,
                root,
                revision=0,
                previous=None,
                start=0,
                pairs=((0, Lang.EN, Lang.ZH, True),),
                total_sources=1,
            )
            live = LiveS2STDataset(root, "lineage", poll_seconds=0.01, status_seconds=1)

            with self.assertRaisesRegex(ValueError, "not in the catalog"):
                live.load_state_dict(
                    {
                        "lineage_id": "lineage",
                        "snapshot_id": "missing",
                        "pair_cursor": 0,
                        "view_id": live.state_dict()["view_id"],
                    }
                )
            with self.assertRaisesRegex(ValueError, "beyond its snapshot prefix"):
                live.load_state_dict(
                    {
                        "lineage_id": "lineage",
                        "snapshot_id": "tts-0",
                        "pair_cursor": 2,
                        "view_id": live.state_dict()["view_id"],
                    }
                )
            live.close()


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
):
    store = root / "stores" / f"tts-{revision}"
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
        stage=S2STStage.TTS,
        snapshot_id=f"tts-{revision}",
        upstream_snapshot_id=f"translation-{revision}",
        upstream_digest=f"translation-digest-{revision}",
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


if __name__ == "__main__":
    unittest.main()
