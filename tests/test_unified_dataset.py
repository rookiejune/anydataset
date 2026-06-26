import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioReq,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextView,
    ViewRef,
)
from anydataset.store import DatasetWriter, ViewInput, ViewMaterializer
from anydataset.store.reader import read_store_dataset


class UnifiedDatasetSourceTest(unittest.TestCase):
    def test_anydataset_reads_waveform_dataset_written_by_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(
                        waveform=waveform,
                        duration=0.75,
                        label="speech",
                        text="hello",
                    )
                ]
            )

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output), split="train"),
                cache_root=root / "cache",
            )
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM], waveform))
        self.assertEqual(audio.required[AudioKey.SAMPLE_RATE], 4)
        self.assertEqual(audio.optional[AudioOptKey.DURATION], 0.75)
        self.assertEqual(audio.optional[AudioOptKey.LABEL], "speech")
        self.assertEqual(text.views[TextView.TEXT], "hello")

    def test_file_view_is_extracted_to_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output)),
                cache_root=root / "cache",
            )
            sample = dataset[0]

            file_view = Path(
                sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
            )

            self.assertTrue(file_view.is_file())
            self.assertEqual(file_view.read_bytes(), b"RIFF-data")
            self.assertTrue(str(file_view).startswith(str(root / "cache")))

    def test_reader_loads_payloads_across_writer_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            samples = [
                _audio_sample(torch.tensor([[float(index)]]), sample_rate=4)
                for index in range(3)
            ]
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=1,
            ).write(samples)

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output)),
                cache_root=root / "cache",
            )
            values = [
                sample[Role.DEFAULT, Modality.AUDIO]
                .views[AudioView.WAVEFORM]
                .item()
                for sample in dataset
            ]

        self.assertEqual(values, [0.0, 1.0, 2.0])

    def test_reader_loads_dataset_with_only_longcat_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform)]
            )

            def provider(view: ViewInput):
                return {"semantic_codes": view.value.to(torch.int64) + 10}

            ViewMaterializer(
                input_dir=source,
                output_dir=target,
                input_ref=ViewRef(Modality.AUDIO, AudioView.WAVEFORM),
                output_ref=ViewRef(Modality.AUDIO, AudioView.LONGCAT),
                transform=provider,
                provider_name="toy_longcat",
                provider_version="1",
            ).write()

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(target), split="train"),
                cache_root=root / "cache",
            )
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.LONGCAT})
        self.assertTrue(
            torch.equal(
                audio.views[AudioView.LONGCAT]["semantic_codes"],
                torch.tensor([[11, 12, 13]]),
            )
        )

    def test_schema_selects_requested_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output)),
                cache_root=root / "cache",
            )
            sample = dataset[0]
            schema = {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.FILE}),
                    required=frozenset({AudioKey.SAMPLE_RATE}),
                )
            }

            resolved = AnyDataset.resolve_sample(sample, schema)

        audio = resolved[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.FILE})
        self.assertEqual(audio.required[AudioKey.SAMPLE_RATE], 16000)

    def test_reader_can_load_selected_view_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="multi-view").write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[1.0, 2.0]]),
                        file=str(source),
                        sample_rate=16000,
                    )
                ]
            )
            ref = ViewRef(Modality.AUDIO, AudioView.FILE)

            dataset = read_store_dataset(
                output,
                cache_path=root / "cache",
                views=(ref,),
            )

        self.assertEqual(set(dataset.views), {ref})
        self.assertEqual(len(dataset.samples), 1)

    def test_schema_missing_view_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output)),
                cache_root=root / "cache",
            )
            sample = dataset[0]
            schema = {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.WAVEFORM}),
                    required=frozenset({AudioKey.SAMPLE_RATE}),
                )
            }

            with self.assertRaises(KeyError):
                AnyDataset.resolve_sample(sample, schema)

    def test_reader_rejects_missing_selected_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=b"RIFF-data", sample_rate=16000)]
            )

            with self.assertRaises(KeyError):
                read_store_dataset(
                    output,
                    cache_path=root / "cache",
                    views=(ViewRef(Modality.AUDIO, AudioView.WAVEFORM),),
                )

    def test_sharded_iteration_is_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            samples = [
                _audio_sample(torch.tensor([[float(index)]]), sample_rate=4)
                for index in range(4)
            ]
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(samples)
            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output), split="train"),
                cache_root=root / "cache",
            )

            shard_zero = [
                sample[Role.DEFAULT, Modality.AUDIO]
                .views[AudioView.WAVEFORM]
                .item()
                for sample in dataset.iter_shard(2, 0)
            ]
            shard_one = [
                sample[Role.DEFAULT, Modality.AUDIO]
                .views[AudioView.WAVEFORM]
                .item()
                for sample in dataset.iter_shard(2, 1)
            ]

        self.assertEqual(shard_zero, [0.0, 2.0])
        self.assertEqual(shard_one, [1.0, 3.0])


def _audio_sample(
    waveform=None,
    *,
    file=None,
    sample_rate: int = 4,
    duration=None,
    label=None,
    text: str | None = None,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = waveform
    if file is not None:
        views[AudioView.FILE] = file
    optional = {}
    if duration is not None:
        optional[AudioOptKey.DURATION] = duration
    if label is not None:
        optional[AudioOptKey.LABEL] = label
    sample = {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
            required={AudioKey.SAMPLE_RATE: sample_rate},
            optional=optional,
        )
    }
    if text is not None:
        sample[(Role.DEFAULT, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: text}
        )
    return sample


if __name__ == "__main__":
    unittest.main()
