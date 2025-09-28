"""
General-purpose dataset analysis utilities for FinData Cleaner.

Main entry point for Streamlit:
    from analysis import describe_table_basic, profile_dataframe

1) Quick shape/memory + head:
    info = describe_table_basic(df)
    # info = {"rows": ..., "cols": ..., "memory_mb": ..., "columns": [..names..]}

2) Column-wise profiles (safe, no means/medians):
    prof = profile_dataframe(df, max_categorical_unique=20)
    # prof is a list[ColumnProfile dict], each with fields described below.

Design goals:
- Robust to messy CSV/TXT (object dtype, números com vírgula, datas variadas).
- Only lightweight conversions for inference (não altera o df original).
- “Finite categorical” = nº de únicos <= max_categorical_unique (default=20).
- For discrete columns: list sorted uniques if finite (up to 20),
  plus 3 primeiros e 3 últimos; for large-cardinality, mostram-se top 10.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers: text/encoding
# -------------------------

def _clean_unicode(s: str) -> str:
    # Normaliza acentos e remove NBSP; não altera significado
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s)).replace("\xa0", " ").strip()


# -------------------------
# Type inference (safe)
# -------------------------

_NUM_RE = re.compile(r"^-?\d+(?:[\.,]\d+)?$")

def _try_parse_numeric(series: pd.Series) -> Tuple[Optional[pd.Series], float]:
    """
    Try to parse a text/object series into numeric, handling pt-BR and en-US styles.
    Returns (parsed_series_or_None, parsed_ratio).
    """
    s = series.dropna().astype(str).str.strip()

    if len(s) == 0:
        return None, 0.0

    # Heuristic 1: quick numeric-ish mask
    mask_basic = s.str.match(_NUM_RE)
    frac_basic = mask_basic.mean()

    # Strategy A: pt-BR (1.234,56) -> remove thousands dot and replace comma with dot
    sabr = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    a_num = pd.to_numeric(sabr, errors="coerce")
    frac_a = a_num.notna().mean()

    # Strategy B: en-US (1,234.56) -> remove thousands comma
    sb = s.str.replace(r",(?=\d{3}(?:\D|$))", "", regex=True)
    b_num = pd.to_numeric(sb, errors="coerce")
    frac_b = b_num.notna().mean()

    # Pick the better parse
    if max(frac_a, frac_b, frac_basic) < 0.7:
        return None, max(frac_a, frac_b, frac_basic)  # not confident

    if frac_a >= frac_b:
        return a_num, frac_a
    return b_num, frac_b


def _try_parse_dates(series: pd.Series) -> Tuple[Optional[pd.Series], float]:
    """
    Try to parse to datetime with multiple settings.
    Returns (parsed_series_or_None, parsed_ratio).
    """
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return None, 0.0

    for dayfirst in (True, False):
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
        frac = parsed.notna().mean()
        if frac >= 0.7:
            return parsed, frac

    return None, 0.0


def infer_series_type(series: pd.Series) -> str:
    """
    Returns one of: 'numeric', 'date', 'categorical', 'text'
    """
    # Non-object dtypes shortcut
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Object column: attempt to parse into numeric or date
    num_parsed, num_ratio = _try_parse_numeric(series)
    if num_parsed is not None and num_ratio >= 0.7:
        return "numeric"

    dt_parsed, dt_ratio = _try_parse_dates(series)
    if dt_parsed is not None and dt_ratio >= 0.7:
        return "date"

    # Decide between categorical vs text:
    # heuristic: small number of unique (<= 20 by default in profile step) => categorical
    # here we return 'text' and let the caller decide if it's finite categorical
    return "text"


# -------------------------
# Column profile dataclass
# -------------------------

@dataclass
class ColumnProfile:
    name: str
    pandas_dtype: str
    inferred_type: str          # numeric | date | categorical | text
    non_null: int
    nulls: int
    unique: int
    sample_values: List[str]    # up to 6 examples (3 first + 3 last of uniques sorted)
    finite_values: Optional[List[str]]  # all uniques (sorted) if unique <= max_categorical_unique
    top_values: Optional[List[Tuple[str, int]]]  # top 10 if high cardinality text/categorical
    min_value: Optional[str]    # for numeric/date
    max_value: Optional[str]    # for numeric/date


def _examples_from_uniques(u: List[Any]) -> List[str]:
    # first 3 + last 3 unique, stringified
    if not u:
        return []
    if len(u) <= 6:
        return [str(x) for x in u]
    return [str(x) for x in (u[:3] + u[-3:])]


# -------------------------
# Public API
# -------------------------

def describe_table_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight info for header section (rows/cols/memory/columns)."""
    mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "memory_mb": round(mem_mb, 3),
        "columns": list(map(str, df.columns)),
    }


def profile_dataframe(
    df: pd.DataFrame,
    max_categorical_unique: int = 20,
) -> List[Dict[str, Any]]:
    """
    Column-wise profiles. No averages/medians; only safe stats.

    - numeric/date => min/max
    - finite categorical (unique <= max_categorical_unique) => list all values (sorted),
      plus 3 primeiros/3 últimos exemplos
    - large-cardinality categorical/text => top 10 valores (com contagem)
    - all => nulls, non_null, unique

    Returns a list of dicts (serialized ColumnProfile).
    """
    profiles: List[ColumnProfile] = []

    for col in df.columns:
        s = df[col]
        pandas_dtype = str(s.dtype)
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        unique_ct = int(s.nunique(dropna=True))

        inferred = infer_series_type(s)

        # Prepare values/uniques sorted (stringified)
        uniques_sorted: List[str] = []
        try:
            uniques_sorted = sorted({_clean_unicode(x) for x in s.dropna().astype(str).unique()})
        except Exception:
            # Fallback if sorting fails because of mixed types
            uniques_sorted = [ _clean_unicode(str(x)) for x in pd.unique(s.dropna().astype(str)) ]

        sample_values = _examples_from_uniques(uniques_sorted)

        finite_values: Optional[List[str]] = None
        top_values: Optional[List[Tuple[str, int]]] = None
        min_value: Optional[str] = None
        max_value: Optional[str] = None

        if inferred == "numeric":
            # Parse numeric for min/max
            if pd.api.types.is_numeric_dtype(s):
                num = s.astype("float64")
            else:
                num, ratio = _try_parse_numeric(s)
                if num is None:
                    # Could not parse confidently; treat as text/categorical
                    inferred = "text"
                    num = None

            if inferred == "numeric" and num is not None:
                if num.notna().any():
                    min_value = str(np.nanmin(num.values))
                    max_value = str(np.nanmax(num.values))

        elif inferred == "date":
            if pd.api.types.is_datetime64_any_dtype(s):
                dt = s
            else:
                dt, ratio = _try_parse_dates(s)
            if dt is not None and dt.notna().any():
                min_value = str(pd.to_datetime(dt.min()).date())
                max_value = str(pd.to_datetime(dt.max()).date())

        # Decide categorical vs text and fill finite/top lists
        if inferred in ("text", "categorical"):
            if unique_ct <= max_categorical_unique:
                inferred = "categorical"
                finite_values = uniques_sorted  # show all values (<= 20)
            else:
                # top 10 most frequent
                vc = (
                    s.astype(str)
                    .apply(_clean_unicode)
                    .value_counts(dropna=True)
                    .head(10)
                )
                top_values = [(str(idx), int(cnt)) for idx, cnt in vc.items()]

        profiles.append(
            ColumnProfile(
                name=str(col),
                pandas_dtype=pandas_dtype,
                inferred_type=inferred,
                non_null=non_null,
                nulls=nulls,
                unique=unique_ct,
                sample_values=sample_values,
                finite_values=finite_values,
                top_values=top_values,
                min_value=min_value,
                max_value=max_value,
            )
        )

    return [asdict(p) for p in profiles]


# -------------------------
# Convenience for Streamlit (optional)
# -------------------------

def profiles_to_dataframe(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten profiles to a compact DataFrame for display:
    columns: [name, inferred_type, pandas_dtype, non_null, nulls, unique, min, max, examples]
    """
    rows = []
    for p in profiles:
        examples = ", ".join(p["sample_values"]) if p.get("sample_values") else ""
        rng = (p.get("min_value") or "", p.get("max_value") or "")
        rows.append(
            {
                "coluna": p["name"],
                "tipo_inferido": p["inferred_type"],
                "dtype_pandas": p["pandas_dtype"],
                "não_nulos": p["non_null"],
                "nulos": p["nulls"],
                "únicos": p["unique"],
                "min": rng[0],
                "max": rng[1],
                "exemplos(≤6)": examples,
            }
        )
    return pd.DataFrame(rows)


def finite_values_as_dict(profiles: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Returns {column: finite_values[]} for columns with finite categorical values.
    """
    out = {}
    for p in profiles:
        if p["inferred_type"] == "categorical" and p.get("finite_values"):
            out[p["name"]] = p["finite_values"]
    return out
