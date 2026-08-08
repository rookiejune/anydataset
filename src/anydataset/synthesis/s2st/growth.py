from __future__ import annotations

import hashlib
from dataclasses import replace

from .model import (
    GrowthPhase,
    GrowthPlan,
    LanguageCatchup,
    PairKey,
    PairPlan,
    ReferenceAudio,
    ReferenceVoice,
    S2STConfig,
    S2STState,
    SlotCursor,
    SourceFamily,
    SourceKey,
    SpeakerList,
    SpeakerVoice,
    validate_successor,
)


def plan_growth(
    config: S2STConfig,
    state: S2STState | None = None,
) -> GrowthPlan:
    """Plan one append-only source-family revision."""

    if not isinstance(config, S2STConfig):
        raise TypeError("config must be an S2STConfig.")
    initial = state is None
    current = S2STState.empty(config) if state is None else _updated(config, state)
    limit = config.growth.initial_sources if initial else config.growth.interval_sources

    backfill = _backfill(config, current, limit)
    if backfill:
        current = replace(current, pairs=(*current.pairs, *backfill))
        return _commit(current, GrowthPhase.LANGUAGE_BACKFILL, (), backfill)

    caught_up, current = _catchup(config, current, limit)
    if caught_up:
        return _commit(current, GrowthPhase.LANGUAGE_SOURCES, caught_up[0], caught_up[1])

    added_families, added_pairs, current = _admit(
        config,
        current,
        limit,
        allowed_languages=None,
    )
    if added_families:
        phase = GrowthPhase.INITIAL if initial else GrowthPhase.VERTICAL
        return _commit(current, phase, added_families, added_pairs)
    return GrowthPlan(
        phase=GrowthPhase.IDLE,
        revision=None,
        added_families=(),
        added_pairs=(),
        state=current,
    )


def _updated(config: S2STConfig, state: S2STState) -> S2STState:
    if not isinstance(state, S2STState):
        raise TypeError("state must be an S2STState or None.")
    if state.lineage_id != config.lineage_id:
        raise ValueError("S2ST model or voice identity changed; start a new lineage.")
    declaration = config.declaration
    validate_successor(state.declaration, declaration)
    if declaration == state.declaration:
        return state

    old_languages = tuple(item.language for item in state.declaration.languages)
    new_languages = tuple(item.language for item in declaration.languages[len(old_languages) :])
    counts = _language_counts(state)
    target = max((counts.get(language, 0) for language in old_languages), default=0)
    catchup = [*state.catchup]
    catchup.extend(LanguageCatchup(language, target) for language in new_languages)

    cursor_names = {cursor.slot for cursor in state.cursors}
    cursors = [*state.cursors]
    cursors.extend(
        SlotCursor(source.name)
        for source in config.slots
        if source.name not in cursor_names
    )
    speaker_pool_revision = state.speaker_pool_revision
    if declaration.speakers != state.declaration.speakers:
        speaker_pool_revision += 1
    old_slot_names = tuple(
        source.name
        for language in state.declaration.languages
        for source in language.sources
    )
    new_slot_names = tuple(source.name for source in config.slots)
    next_slot_index = state.next_slot_index
    if old_slot_names:
        next_name = old_slot_names[state.next_slot_index % len(old_slot_names)]
        next_slot_index = new_slot_names.index(next_name)
    return replace(
        state,
        declaration=declaration,
        speaker_pool_revision=speaker_pool_revision,
        cursors=tuple(cursors),
        catchup=tuple(catchup),
        next_slot_index=next_slot_index,
    )


def _backfill(
    config: S2STConfig,
    state: S2STState,
    limit: int,
) -> tuple[PairPlan, ...]:
    existing = {pair.key for pair in state.pairs}
    languages = tuple(language.language for language in config.languages)
    added: list[PairPlan] = []
    touched = 0
    for family in state.families:
        missing = [
            language
            for language in languages
            if language is not family.language
            and PairKey(family.key, language) not in existing
        ]
        if not missing:
            continue
        if touched >= limit:
            break
        touched += 1
        for language in missing:
            pair = PairPlan(
                key=PairKey(family.key, language),
                source_sequence=family.sequence,
                first_for_source=False,
            )
            added.append(pair)
            existing.add(pair.key)
    return tuple(added)


def _catchup(
    config: S2STConfig,
    state: S2STState,
    limit: int,
) -> tuple[tuple[tuple[SourceFamily, ...], tuple[PairPlan, ...]] | None, S2STState]:
    if not state.catchup:
        return None, state
    counts = _language_counts(state)
    pending = tuple(
        item
        for item in state.catchup
        if counts.get(item.language, 0) < item.source_target
        and not _language_exhausted(config, state, item.language)
    )
    if not pending:
        return None, replace(state, catchup=())
    allowed = frozenset(item.language for item in pending)
    families, pairs, updated = _admit(
        config,
        state,
        limit,
        allowed_languages=allowed,
        source_targets={item.language: item.source_target for item in pending},
    )
    counts = _language_counts(updated)
    remaining = tuple(
        item
        for item in pending
        if counts.get(item.language, 0) < item.source_target
        and not _language_exhausted(config, updated, item.language)
    )
    updated = replace(updated, catchup=remaining)
    if not families:
        return None, updated
    return (families, pairs), updated


def _admit(
    config: S2STConfig,
    state: S2STState,
    limit: int,
    *,
    allowed_languages,
    source_targets=None,
) -> tuple[tuple[SourceFamily, ...], tuple[PairPlan, ...], S2STState]:
    slots = tuple(
        (language.language, source)
        for language in config.languages
        for source in language.sources
    )
    if not slots:
        return (), (), state
    cursors = {cursor.slot: cursor for cursor in state.cursors}
    families = [*state.families]
    pairs = [*state.pairs]
    added_families: list[SourceFamily] = []
    added_pairs: list[PairPlan] = []
    next_slot = state.next_slot_index % len(slots)
    stalled = 0
    lengths: dict[str, int] = {}
    sealed_sources: dict[str, bool] = {}

    while len(added_families) < limit and stalled < len(slots):
        language, slot = slots[next_slot]
        next_slot = (next_slot + 1) % len(slots)
        if allowed_languages is not None and language not in allowed_languages:
            stalled += 1
            continue
        if source_targets is not None:
            count = sum(family.language is language for family in families)
            if count >= source_targets[language]:
                stalled += 1
                continue
        cursor = cursors[slot.name]
        length = lengths.get(slot.name)
        if length is None:
            dataset = slot.dataset()
            from ...dataset import MapStyleABC

            if not isinstance(dataset, MapStyleABC):
                raise TypeError(
                    f"S2ST source slot {slot.name!r} factory must return MapStyleABC."
                )
            try:
                length = len(dataset)
                sealed = _dataset_is_sealed(dataset, slot.name)
            finally:
                _close_dataset(dataset)
            lengths[slot.name] = length
            sealed_sources[slot.name] = sealed
        else:
            sealed = sealed_sources[slot.name]
        if cursor.sample_count is not None and length < cursor.sample_count:
            raise ValueError(
                f"S2ST source slot {slot.name!r} shrank within one lineage: "
                f"{cursor.sample_count} -> {length}."
            )
        if cursor.next_row >= length:
            cursors[slot.name] = replace(
                cursor,
                sample_count=length,
                exhausted=sealed,
            )
            stalled += 1
            continue

        key = SourceKey(slot.name, cursor.next_row)
        family = SourceFamily(
            key=key,
            sequence=len(families),
            language=language,
            voice=_voice(config, state, key),
        )
        targets = [
            candidate.language
            for candidate in config.languages
            if candidate.language is not language
        ]
        family_pairs = [
            PairPlan(
                key=PairKey(key, target),
                source_sequence=family.sequence,
                first_for_source=index == 0,
            )
            for index, target in enumerate(targets)
        ]
        families.append(family)
        pairs.extend(family_pairs)
        added_families.append(family)
        added_pairs.extend(family_pairs)
        next_row = cursor.next_row + 1
        cursors[slot.name] = replace(
            cursor,
            next_row=next_row,
            sample_count=length,
            exhausted=sealed and next_row >= length,
        )
        stalled = 0

    updated = replace(
        state,
        families=tuple(families),
        pairs=tuple(pairs),
        cursors=tuple(cursors[cursor.slot] for cursor in state.cursors),
        next_slot_index=next_slot,
    )
    return tuple(added_families), tuple(added_pairs), updated


def _voice(config: S2STConfig, state: S2STState, key: SourceKey):
    if isinstance(config.voice, ReferenceAudio):
        return ReferenceVoice(key)
    if not isinstance(config.voice, SpeakerList):
        raise TypeError("unsupported S2ST voice configuration.")
    payload = (
        f"{config.voice.seed}:{state.speaker_pool_revision}:{key.slot}:{key.row}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return SpeakerVoice(
        speaker_id=config.voice.speakers[value % len(config.voice.speakers)],
        pool_revision=state.speaker_pool_revision,
    )


def _commit(
    state: S2STState,
    phase: GrowthPhase,
    families: tuple[SourceFamily, ...],
    pairs: tuple[PairPlan, ...],
) -> GrowthPlan:
    revision = state.revision + 1
    updated = replace(state, revision=revision)
    return GrowthPlan(
        phase=phase,
        revision=revision,
        added_families=families,
        added_pairs=pairs,
        state=updated,
    )


def _language_counts(state: S2STState):
    counts = {}
    for family in state.families:
        counts[family.language] = counts.get(family.language, 0) + 1
    return counts


def _language_exhausted(config: S2STConfig, state: S2STState, language) -> bool:
    slots = next(item.sources for item in config.languages if item.language is language)
    cursors = {cursor.slot: cursor for cursor in state.cursors}
    return all(cursors[slot.name].exhausted for slot in slots)


def _close_dataset(dataset) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    prepared = getattr(dataset, "dataset", None)
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


def _dataset_is_sealed(dataset, slot: str) -> bool:
    sealed = getattr(dataset, "sealed", None)
    if sealed is None:
        return True
    if type(sealed) is not bool:
        raise TypeError(
            f"S2ST source slot {slot!r} sealed must be a boolean when provided."
        )
    return sealed


__all__ = ["plan_growth"]
