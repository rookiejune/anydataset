import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioView,
    DatasetWriter,
    ImageItem,
    ImageOptKey,
    ImageView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextOptKey,
    TextView,
    ViewMaterializer,
    ViewRef,
)
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import (
    DatasetManifest,
    SampleItemEntry,
    SampleManifestEntry,
)
from anydataset.store.manifestio import (
    read_samples_manifest,
    read_view_manifest,
)
from anydataset.store.paths import (
    dataset_json_path,
)


class CanonicalStoreTest(unittest.TestCase):
    def test_writer_round_trips_multimodal_canonical_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            waveform = torch.tensor([[1.0, 2.0]])

            DatasetWriter(output, dataset_id="toy", manifest_format="jsonl").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: waveform,
                                AudioView.FILE: str(source),
                            },
                            required={AudioKey.SAMPLE_RATE: 16000},
                            optional={
                                AudioOptKey.DURATION: 0.5,
                                AudioOptKey.LABELS: {"kind": "speech"},
                            },
                        ),
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: [[1, 2], [3, 4]]},
                            optional={ImageOptKey.LABEL: 7},
                        ),
                        (Role.SOURCE, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            optional={TextOptKey.LANG: "en"},
                        ),
                    }
                ]
            )

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(output), split=None),
                cache_root=root / "cache",
            )
            sample = dataset[0]

            audio = sample[Role.DEFAULT, Modality.AUDIO]
            image = sample[Role.DEFAULT, Modality.IMAGE]
            text = sample[Role.SOURCE, Modality.TEXT]
            self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM], waveform))
            self.assertEqual(Path(audio.views[AudioView.FILE]).read_bytes(), b"RIFF-data")
            self.assertEqual(audio.required[AudioKey.SAMPLE_RATE], 16000)
            self.assertEqual(audio.optional[AudioOptKey.LABELS], {"kind": "speech"})
            self.assertTrue(torch.equal(image.views[ImageView.PIXEL], torch.tensor([[1, 2], [3, 4]])))
            self.assertEqual(image.optional[ImageOptKey.LABEL], 7)
            self.assertEqual(text.views[TextView.TEXT], "hello")
            self.assertEqual(text.optional[TextOptKey.LANG], "en")

            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(output)))
            refs = {selection.ref for selection in manifest.views}
            self.assertIn(ViewRef(Modality.AUDIO, AudioView.WAVEFORM), refs)
            self.assertIn(ViewRef(Modality.AUDIO, AudioView.FILE), refs)
            self.assertIn(ViewRef(Modality.IMAGE, ImageView.PIXEL), refs)
            self.assertIn(ViewRef(Modality.TEXT, TextView.TEXT, role=Role.SOURCE), refs)

            entry = next(read_samples_manifest(output))
            audio_entry = entry.item((Role.DEFAULT, Modality.AUDIO))
            self.assertIsNotNone(audio_entry)
            self.assertEqual(audio_entry.required[AudioKey.SAMPLE_RATE], 16000)
            self.assertEqual(audio_entry.optional[AudioOptKey.DURATION], 0.5)

    def test_manifest_entries_store_item_attrs_by_reference(self):
        entry = SampleManifestEntry(
            sample_id="sample-0",
            dataset_name="toy",
            items=(
                SampleItemEntry(
                    ref=(Role.SOURCE, Modality.TEXT),
                    optional={TextOptKey.LANG: "en"},
                ),
            ),
        )

        loaded = SampleManifestEntry.from_dict(entry.to_dict())

        self.assertEqual(loaded, entry)
        self.assertEqual(loaded.items[0].optional, {"lang": "en"})

    def test_materializer_can_add_text_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy", manifest_format="jsonl").write(
                [
                    {
                        (Role.SOURCE, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"}
                        )
                    }
                ]
            )
            input_ref = ViewRef(Modality.TEXT, TextView.TEXT, role=Role.SOURCE)
            output_ref = ViewRef(Modality.TEXT, TextView.TEXT, role=Role.TARGET)

            ViewMaterializer(
                input_dir=source,
                output_dir=target,
                input_ref=input_ref,
                output_ref=output_ref,
                transform=lambda view: view.value.upper(),
                provider_name="toy_upper",
                provider_version="1",
            ).write()

            entry = next(read_view_manifest(target, output_ref, next(
                selection.revision
                for selection in DatasetManifest.from_dict(
                    read_json(dataset_json_path(target))
                ).views
                if selection.ref == output_ref
            )))
            self.assertEqual(entry.dtype, "text")


if __name__ == "__main__":
    unittest.main()
