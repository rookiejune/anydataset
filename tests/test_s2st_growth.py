import unittest

from anydataset.dataset import MapStyleABC
from anydataset.synthesis.s2st import (
    Growth,
    GrowthPhase,
    LanguageSources,
    ReferenceAudio,
    S2STConfig,
    S2STState,
    SourceKey,
    SourceSlot,
    SpeakerList,
    SpeakerVoice,
    plan_growth,
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


class S2STGrowthTest(unittest.TestCase):
    def test_repeated_physical_source_slots_keep_independent_cursors(self):
        dataset = _Dataset(Lang.EN, 2)
        config = _config(
            (
                _language(Lang.EN, ("en-a", dataset), ("en-repeat", dataset)),
                _language(Lang.ZH, ("zh", _Dataset(Lang.ZH, 2))),
            ),
            initial=3,
            interval=3,
        )

        plan = plan_growth(config)

        self.assertEqual(plan.phase, GrowthPhase.INITIAL)
        self.assertEqual(
            [family.key for family in plan.added_families],
            [SourceKey("en-a", 0), SourceKey("en-repeat", 0), SourceKey("zh", 0)],
        )

    def test_new_language_backfills_old_sources_before_admitting_new_sources(self):
        first = _config(
            (
                _language(Lang.EN, ("en", _Dataset(Lang.EN, 3))),
                _language(Lang.ZH, ("zh", _Dataset(Lang.ZH, 3))),
            ),
            initial=4,
            interval=2,
        )
        initial = plan_growth(first)
        expanded = _config(
            (
                *first.languages,
                _language(Lang.FR, ("fr", _Dataset(Lang.FR, 4))),
            ),
            initial=4,
            interval=2,
        )

        backfill = plan_growth(expanded, initial.state)
        next_backfill = plan_growth(expanded, backfill.state)
        catchup = plan_growth(expanded, next_backfill.state)

        self.assertEqual(backfill.phase, GrowthPhase.LANGUAGE_BACKFILL)
        self.assertFalse(backfill.added_families)
        self.assertEqual(
            [pair.key.target_language for pair in backfill.added_pairs],
            [Lang.FR, Lang.FR],
        )
        self.assertEqual(next_backfill.phase, GrowthPhase.LANGUAGE_BACKFILL)
        self.assertEqual(catchup.phase, GrowthPhase.LANGUAGE_SOURCES)
        self.assertTrue(all(family.language is Lang.FR for family in catchup.added_families))

    def test_added_speaker_affects_future_families_only(self):
        config = _config(
            (
                _language(Lang.EN, ("en", _Dataset(Lang.EN, 4))),
                _language(Lang.ZH, ("zh", _Dataset(Lang.ZH, 4))),
            ),
            initial=2,
            interval=2,
            speakers=("Vivian",),
        )
        first = plan_growth(config)
        expanded = _config(
            config.languages,
            initial=2,
            interval=2,
            speakers=("Vivian", "Ryan"),
        )

        second = plan_growth(expanded, first.state)

        old = [family.voice for family in second.state.families[:2]]
        new = [family.voice for family in second.added_families]
        self.assertEqual(
            old,
            [SpeakerVoice("Vivian", 0), SpeakerVoice("Vivian", 0)],
        )
        self.assertTrue(all(isinstance(voice, SpeakerVoice) for voice in new))
        self.assertTrue(all(voice.pool_revision == 1 for voice in new))

    def test_state_round_trip_preserves_growth_identity(self):
        config = _config(
            (
                _language(Lang.EN, ("en", _Dataset(Lang.EN, 2))),
                _language(Lang.ZH, ("zh", _Dataset(Lang.ZH, 2))),
            ),
            initial=2,
            interval=1,
        )
        state = plan_growth(config).state

        restored = S2STState.from_dict(state.to_dict())

        self.assertEqual(restored, state)

    def test_reference_mode_requires_audio_and_loads_declared_source(self):
        dataset = _Dataset(Lang.EN, 1, audio=True)
        slot = SourceSlot(
            "en",
            "dataset-en",
            lambda: dataset,
            text=(Role.DEFAULT, Modality.TEXT),
            audio=(Role.DEFAULT, Modality.AUDIO),
        )
        config = S2STConfig(
            name="unit",
            languages=(
                LanguageSources(Lang.EN, (slot,)),
                _language(Lang.ZH, ("zh", _Dataset(Lang.ZH, 1, audio=True)), audio=True),
            ),
            translator_id="translator-v1",
            tts_id="tts-v1",
            voice=ReferenceAudio(),
            growth=Growth(1, 1),
        )

        source = config.source(SourceKey("en", 0))

        self.assertEqual(source.language, Lang.EN)
        self.assertEqual(source.text.meta[TextMeta.LANG], Lang.EN)
        self.assertIsInstance(source.audio, AudioItem)


class _Dataset(MapStyleABC):
    def __init__(self, language, count, *, audio=False):
        self.language = language
        self.count = count
        self.audio = audio

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index < 0 or index >= self.count:
            raise IndexError(index)
        sample = {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"{self.language.value}-{index}"},
                meta={TextMeta.LANG: self.language},
            )
        }
        if self.audio:
            sample[(Role.DEFAULT, Modality.AUDIO)] = AudioItem(
                views={AudioView.WAVEFORM: (__import__("torch").zeros(1, 4), 16000)}
            )
        return sample


def _language(language, *sources, audio=False):
    return LanguageSources(
        language,
        tuple(
            SourceSlot(
                name,
                f"dataset-{name}",
                lambda dataset=dataset: dataset,
                audio=(Role.DEFAULT, Modality.AUDIO) if audio else None,
            )
            for name, dataset in sources
        ),
    )


def _config(languages, *, initial, interval, speakers=("Vivian", "Ryan")):
    return S2STConfig(
        name="unit",
        languages=tuple(languages),
        translator_id="translator-v1",
        tts_id="tts-v1",
        voice=SpeakerList(tuple(speakers), seed=7),
        growth=Growth(initial, interval),
    )


if __name__ == "__main__":
    unittest.main()
