import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    ImageItem,
    ImageView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.reader import read_store_dataset


class StoreCanonicalTest(unittest.TestCase):
    def test_writer_round_trips_canonical_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            audio_file = root / "audio.wav"
            audio_file.write_bytes(b"RIFF-data")
            DatasetWriter(output, dataset_id="toy").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4),
                                AudioView.FILE: str(audio_file),
                            }
                        ),
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: [[1, 2], [3, 4]]}
                        ),
                        (Role.SOURCE, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"}
                        ),
                    }
                ]
            )

            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output)),
                cache_root=root / "cache",
            )
            sample = dataset[0]
            stored = read_store_dataset(output)

        self.assertEqual(
            set(stored.views),
            {
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
                (Role.DEFAULT, Modality.AUDIO, AudioView.FILE),
                (Role.DEFAULT, Modality.IMAGE, ImageView.PIXEL),
                (Role.SOURCE, Modality.TEXT, TextView.TEXT),
            },
        )
        waveform, sample_rate = sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]
        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(sample_rate, 4)
        self.assertTrue(
            torch.equal(
                sample[Role.DEFAULT, Modality.IMAGE].views[ImageView.PIXEL],
                torch.tensor([[1, 2], [3, 4]]),
            )
        )
        self.assertEqual(sample[Role.SOURCE, Modality.TEXT].views[TextView.TEXT], "hello")

    def test_materializer_can_add_text_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy").write(
                [
                    {
                        (Role.SOURCE, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"}
                        )
                    }
                ]
            )
            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(source)),
                cache_root=root / "cache-source",
            )

            ViewMaterializer(target, dataset_id="toy").write(dataset, _UpperProvider())

            sample = AnyDataset(
                Spec(source=Source.STORE, path=str(target)),
                cache_root=root / "cache-target",
            )[0]
            stored = read_store_dataset(target)

        self.assertEqual(
            set(stored.views),
            {(Role.SOURCE, Modality.TEXT, TextView.TEXT)},
        )
        self.assertEqual(sample[Role.SOURCE, Modality.TEXT].views[TextView.TEXT], "HELLO")


class _UpperProvider:
    output = TextView.TEXT

    def __call__(self, views):
        return views[TextView.TEXT].upper()


if __name__ == "__main__":
    unittest.main()
