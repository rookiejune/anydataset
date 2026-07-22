from __future__ import annotations

from collections.abc import Mapping
from enum import auto

from .._compat import StrEnum


class Lang(StrEnum):
    UND = auto()
    AF = auto()
    AR = auto()
    BE = auto()
    BG = auto()
    CA = auto()
    CS = auto()
    DA = auto()
    DE = auto()
    EN = auto()
    ES = auto()
    ET = auto()
    FA = auto()
    FI = auto()
    FR = auto()
    HE = auto()
    HI = auto()
    HR = auto()
    HU = auto()
    ID = auto()
    IT = auto()
    JA = auto()
    KK = auto()
    KO = auto()
    MK = auto()
    MS = auto()
    NL = auto()
    NO = auto()
    PL = auto()
    PS = auto()
    PT = auto()
    RO = auto()
    RU = auto()
    SK = auto()
    SL = auto()
    SR = auto()
    SV = auto()
    TH = auto()
    TR = auto()
    UK = auto()
    UR = auto()
    VI = auto()
    ZH = auto()


def remap_lang(value: Lang | str, remap: Mapping[str, Lang] | None = None) -> Lang:
    if isinstance(value, Lang):
        return value
    if not isinstance(value, str):
        raise TypeError("language label must be a Lang or string.")

    raw = value.strip()
    if raw == "":
        raise ValueError("language label must not be empty.")

    if remap is not None:
        mapped = _mapped_lang(raw, remap)
        if mapped is not None:
            return mapped

    label = _label(raw)
    try:
        return Lang(label)
    except ValueError:
        base = label.split("-", 1)[0]
        try:
            return Lang(base)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported language label {value!r}; pass remap=... or add Lang."
            ) from exc


def _mapped_lang(raw: str, remap: Mapping[str, Lang]) -> Lang | None:
    for key in (raw, _label(raw)):
        if key not in remap:
            continue
        value = remap[key]
        if not isinstance(value, Lang):
            raise TypeError("language remap values must be Lang values.")
        return value
    return None


def _label(value: str) -> str:
    return value.lower().replace("_", "-")


__all__ = ["Lang", "remap_lang"]
