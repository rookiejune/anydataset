import unittest

from anydataset.dataset import MapStyleABC
from anydataset.synthesis.s2st import (
    Growth,
    LanguageSources,
    S2STConfig,
    S2STLayout,
    S2STView,
    SourceSlot,
    SpeakerList,
    ToyS2STDataset,
)
from anydataset.types import AudioMeta, Lang, Modality, Role


class S2STToyTest(unittest.TestCase):
    def test_toy_pairs_cover_language_matrix_and_keep_family_speaker(self):
        toy = ToyS2STDataset(_config(), sources=3)

        self.assertEqual(len(toy), 6)
        first = toy[0]
        second = toy[1]
        self.assertEqual(
            first[(Role.SOURCE, Modality.AUDIO)].meta[AudioMeta.SPEAKER_ID],
            second[(Role.SOURCE, Modality.AUDIO)].meta[AudioMeta.SPEAKER_ID],
        )
        self.assertEqual(
            first[(Role.SOURCE, Modality.AUDIO)].meta[AudioMeta.SPEAKER_ID],
            first[(Role.TARGET, Modality.AUDIO)].meta[AudioMeta.SPEAKER_ID],
        )

    def test_sources_view_returns_one_row_per_family(self):
        toy = ToyS2STDataset(
            _config(),
            sources=3,
            view=S2STView(layout=S2STLayout.SOURCES),
        )

        self.assertEqual(len(toy), 3)
        self.assertTrue(all(reference[0] is Role.SOURCE for reference in toy[0]))

    def test_toy_round_robin_uses_every_declared_source_slot(self):
        config = _config()
        expanded = S2STConfig(
            name=config.name,
            languages=(
                LanguageSources(
                    Lang.EN,
                    (
                        *config.languages[0].sources,
                        SourceSlot("toy-en-repeat", "toy-en", _EmptyDataset),
                    ),
                ),
                *config.languages[1:],
            ),
            translator_id=config.translator_id,
            tts_id=config.tts_id,
            voice=config.voice,
            growth=config.growth,
        )

        toy = ToyS2STDataset(
            expanded,
            sources=4,
            view=S2STView(source_slots=frozenset(("toy-en-repeat",))),
        )

        self.assertEqual(len(toy), 2)


class _EmptyDataset(MapStyleABC):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def _config():
    return S2STConfig(
        name="toy",
        languages=tuple(
            LanguageSources(
                language,
                (
                    SourceSlot(
                        f"toy-{language.value}",
                        f"toy-{language.value}",
                        _EmptyDataset,
                    ),
                ),
            )
            for language in (Lang.EN, Lang.ZH, Lang.FR)
        ),
        translator_id="translator-v1",
        tts_id="tts-v1",
        voice=SpeakerList(("Vivian", "Ryan"), seed=3),
        growth=Growth(3, 3),
    )


if __name__ == "__main__":
    unittest.main()
