from __future__ import annotations

import pytest

from scripts.check_type_completeness import TypeCounts, _validate_ratchet


def test_type_completeness_ratchet_accepts_improvement() -> None:
    baseline = TypeCounts(known=80, ambiguous=5, unknown=15)

    _validate_ratchet(
        TypeCounts(known=90, ambiguous=4, unknown=6),
        baseline,
    )


@pytest.mark.parametrize(
    "current",
    [
        TypeCounts(known=79, ambiguous=5, unknown=15),
        TypeCounts(known=90, ambiguous=6, unknown=15),
    ],
)
def test_type_completeness_ratchet_rejects_regression(
    current: TypeCounts,
) -> None:
    baseline = TypeCounts(known=80, ambiguous=5, unknown=15)

    with pytest.raises(RuntimeError, match="Type completeness regression"):
        _validate_ratchet(current, baseline)
