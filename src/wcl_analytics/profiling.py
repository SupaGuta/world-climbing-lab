"""Generate `docs/analysis/data-profile.md` from the warehouse.

Descriptive-only inventory: shape, dtypes, completeness, top value_counts,
date min/max, duplicate counts (full + on natural key), plus climbing-specific
extras and cross-view consistency checks. No interpretation, no cleaning, no
modelling — those come later and are tracked separately.

Run with `python -m wcl_analytics.profiling`. Commit the resulting report
alongside this module so it stays version-aligned with the code that
produced it.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import pandas as pd

from wcl_data.config import REPO_ROOT
from wcl_data.exporter import VIEW_NAMES

from .loaders import DEFAULT_DB_PATH, execute_sql, load_view

log = logging.getLogger(__name__)

DEFAULT_REPORT_PATH: Path = REPO_ROOT / "docs" / "analysis" / "data-profile.md"

CATEGORICAL_THRESHOLD = 30
VALUE_COUNTS_TOP_N = 15
ASCENTS_SAMPLE_SIZE = 50_000

# Columns whose contents are date-like (YYYY-MM-DD) and should get min/max.
DATE_LIKE: frozenset[str] = frozenset({
    "date_start", "date_end", "event_date", "birthday",
    "sport_class_review_date", "speed_pb_date", "last_fetched_at", "modified",
})

# Natural keys for duplicate detection, expressed in CSV column space (matches
# `load_view` output). Mirror the DB-enforced UNIQUE constraints in
# `wcl_data.db.schema` where possible; cup_rankings and ascents collapse
# multi-column joins via aliased ids in the view.
NATURAL_KEYS: dict[str, tuple[str, ...]] = {
    "seasons": ("season_ifsc_id",),
    "leagues": ("league_id",),
    "events": ("event_ifsc_id",),
    "competitions": ("event_ifsc_id", "competition_ifsc_id"),
    "athletes": ("athlete_ifsc_id",),
    "cup_rankings": ("athlete_ifsc_id", "cup_ifsc_id", "d_cat_id"),
    "results": ("event_ifsc_id", "discipline", "category", "gender", "athlete_ifsc_id"),
    "round_results": ("category_round_ifsc_id", "athlete_ifsc_id"),
    "ascents": (
        "event_ifsc_id", "discipline", "round_name", "stage_seq",
        "route_ifsc_id", "athlete_ifsc_id",
    ),
}

# Fail-fast: keep NATURAL_KEYS in lockstep with VIEW_NAMES so a new view added
# upstream doesn't crash render_report mid-write with a bare KeyError.
_missing_keys = set(VIEW_NAMES) - set(NATURAL_KEYS)
_extra_keys = set(NATURAL_KEYS) - set(VIEW_NAMES)
if _missing_keys or _extra_keys:
    raise RuntimeError(
        "NATURAL_KEYS out of sync with wcl_data.exporter.VIEW_NAMES: "
        f"missing={sorted(_missing_keys)} extra={sorted(_extra_keys)}"
    )

# Secondary semantic uniqueness check for competitions — flags rows where one
# event has multiple competitions for the same discipline x category x gender.
COMPETITIONS_SEMANTIC_KEY: tuple[str, ...] = (
    "event_ifsc_id", "discipline", "category", "gender",
)

ATHLETES_FILL_COLS: tuple[str, ...] = (
    "height", "arm_span", "birthday", "country_iso3",
    "paraclimbing_sport_class", "sport_class_status", "sport_class_review_date",
    "speed_pb_time", "speed_pb_date",
)

ASCENTS_SCORE_COLS: tuple[str, ...] = ("top", "plus", "time_ms", "zone", "points")

_NULL_SENTINEL = "\x00__WCL_NULL__\x00"


def _h(level: int, text: str) -> str:
    return f"\n{'#' * level} {text}\n\n"


def _escape_pipes(df: pd.DataFrame) -> pd.DataFrame:
    """Escape `|` in object cells/index so markdown tables stay aligned.

    `to_markdown` (via tabulate) does not escape pipes; a single `|` in a
    value silently corrupts the row. Warehouse strings like `round_results.score`
    ('34-  |  28- [5.66]') contain thousands of pipes today.
    """
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].astype(str).str.replace("|", "\\|", regex=False)
    if safe.index.dtype == object:
        safe.index = safe.index.astype(str).str.replace("|", "\\|", regex=False)
    return safe


def _md(df: pd.DataFrame, *, index: bool = True) -> str:
    """Render a DataFrame to GitHub-flavored markdown with pipes escaped."""
    return _escape_pipes(df).to_markdown(index=index)


def _column_summary(df: pd.DataFrame) -> pd.DataFrame:
    nrows = len(df)
    non_null = df.notna().sum()
    if nrows:
        pct = (non_null / nrows * 100).round(2)
    else:
        pct = pd.Series([float("nan")] * len(non_null), index=non_null.index)
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": non_null,
        "non_null_%": pct,
        "n_unique": df.nunique(dropna=True),
    })


def _dup_on_key(df: pd.DataFrame, key: tuple[str, ...]) -> int:
    """Count rows that duplicate on `key`. NULL components collapse to a
    sentinel so `duplicated()`'s NaN-aware comparison doesn't treat them as
    distinct. Raises if any key column is missing from `df` — silent
    return-of-sentinel masked drift in the upstream view shape."""
    missing = [c for c in key if c not in df.columns]
    if missing:
        raise KeyError(
            f"natural-key columns {missing} missing from DataFrame columns "
            f"{list(df.columns)} - NATURAL_KEYS may be out of sync with the view"
        )
    keyed = df[list(key)].fillna(_NULL_SENTINEL)
    return int(keyed.duplicated().sum())


def _profile_view(name: str, df: pd.DataFrame, out: TextIO, *, sampled: bool = False) -> None:
    suffix = f" - sampled (LIMIT {len(df):,})" if sampled else ""
    out.write(_h(2, f"`{name}`{suffix}"))
    out.write(f"- **Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    out.write(f"- **Memory:** {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n\n")

    out.write(_h(3, "Per-column summary"))
    out.write(_md(_column_summary(df)) + "\n\n")

    out.write(_h(3, "describe(include=\"all\")"))
    desc = df.describe(include="all").transpose()
    out.write(_md(desc) + "\n\n")

    out.write(_h(3, "Head (up to 10 rows)"))
    head = df.head(10)
    if head.shape[1] > 12:
        head = head.transpose()
    out.write(_md(head) + "\n\n")

    cat_cols = [
        c for c in df.columns
        if 0 < int(df[c].nunique(dropna=True)) < CATEGORICAL_THRESHOLD
    ]
    if cat_cols:
        out.write(_h(3, f"Top {VALUE_COUNTS_TOP_N} value_counts (low-cardinality columns)"))
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False).head(VALUE_COUNTS_TOP_N)
            out.write(f"**`{col}`**\n\n")
            out.write(_md(vc.to_frame("count")) + "\n\n")

    date_cols = [c for c in df.columns if c in DATE_LIKE]
    if date_cols:
        out.write(_h(3, "Date columns - min/max"))
        for col in date_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                out.write(f"- `{col}`: all NULL\n")
            else:
                out.write(f"- `{col}`: {non_null.min()} -> {non_null.max()} (n={len(non_null):,})\n")
        out.write("\n")

    out.write(_h(3, "Duplicates"))
    sample_note = " _(on sample only - not the full table)_" if sampled else ""
    out.write(f"- Fully-duplicated rows: {int(df.duplicated().sum()):,}{sample_note}\n")
    nat = NATURAL_KEYS[name]
    out.write(f"- Duplicates on natural key `{nat}`: {_dup_on_key(df, nat):,}{sample_note}\n")
    if name == "competitions":
        sec = COMPETITIONS_SEMANTIC_KEY
        out.write(
            f"- Duplicates on semantic key `{sec}`: {_dup_on_key(df, sec):,} "
            "(one event with >1 competition per discipline x category x gender)\n"
        )
    out.write("\n")


def _climbing_extras(
    out: TextIO,
    *,
    db_path: Path,
    competitions: pd.DataFrame,
    athletes: pd.DataFrame,
) -> None:
    out.write(_h(2, "Climbing-specific extras"))

    out.write(_h(3, "Disciplines x categories x genders (from `competitions`)"))
    for col in ("discipline", "category", "gender"):
        vc = competitions[col].value_counts(dropna=False)
        out.write(f"**`{col}`** - {int(competitions[col].nunique(dropna=True))} distinct\n\n")
        out.write(_md(vc.to_frame("count")) + "\n\n")

    out.write(_h(3, "`athletes` fill rates"))
    n = len(athletes)
    counts = [int(athletes[c].notna().sum()) for c in ATHLETES_FILL_COLS]
    fill = pd.DataFrame({
        "non_null": counts,
        "fill_%": [round(c / n * 100, 2) if n else 0.0 for c in counts],
    }, index=list(ATHLETES_FILL_COLS))
    out.write(_md(fill) + "\n\n")
    out.write(f"_n = {n:,} athletes_\n\n")

    out.write(_h(3, "`ascents` per-discipline score-column fill rates (full table)"))
    # LEFT JOIN both legs so orphan ascents (missing competition / discipline)
    # surface as a `<orphan>` row instead of being silently excluded from a
    # 'full table' aggregate.
    sql = """
        SELECT
            COALESCE(d.name, '<orphan>') AS discipline,
            COUNT(*) AS n_rows,
            SUM(CASE WHEN asc_.top IS NOT NULL THEN 1 ELSE 0 END) AS top_filled,
            SUM(CASE WHEN asc_.plus IS NOT NULL THEN 1 ELSE 0 END) AS plus_filled,
            SUM(CASE WHEN asc_.time_ms IS NOT NULL THEN 1 ELSE 0 END) AS time_ms_filled,
            SUM(CASE WHEN asc_.zone IS NOT NULL THEN 1 ELSE 0 END) AS zone_filled,
            SUM(CASE WHEN asc_.points IS NOT NULL THEN 1 ELSE 0 END) AS points_filled
        FROM ascents asc_
        LEFT JOIN competitions c ON asc_.competition_id = c.id
        LEFT JOIN disciplines d ON c.discipline_id = d.id
        GROUP BY COALESCE(d.name, '<orphan>')
        ORDER BY n_rows DESC
    """
    df = execute_sql(sql, db_path=db_path)
    for col in ASCENTS_SCORE_COLS:
        df[f"{col}_%"] = (df[f"{col}_filled"] / df["n_rows"] * 100).round(1)
    display = df[["discipline", "n_rows"] + [f"{c}_%" for c in ASCENTS_SCORE_COLS]]
    out.write(_md(display, index=False) + "\n\n")


def _consistency(out: TextIO, *, db_path: Path) -> None:
    out.write(_h(2, "Cross-view consistency"))

    out.write(_h(3, "Join cardinality"))
    cards = (
        ("results per athlete",
         "SELECT athlete_id, COUNT(*) AS n FROM results GROUP BY athlete_id"),
        ("results per competition",
         "SELECT competition_id, COUNT(*) AS n FROM results GROUP BY competition_id"),
        ("round_results per competition",
         "SELECT competition_id, COUNT(*) AS n FROM round_results GROUP BY competition_id"),
        ("ascents per competition",
         "SELECT competition_id, COUNT(*) AS n FROM ascents GROUP BY competition_id"),
    )
    rows: list[dict[str, object]] = []
    for label, sql in cards:
        df = execute_sql(sql, db_path=db_path)
        if df.empty:
            rows.append({
                "relation": label, "groups": 0, "min": None,
                "median": None, "p95": None, "max": None, "total": 0,
            })
            continue
        s = df["n"]
        rows.append({
            "relation": label,
            "groups": len(df),
            "min": int(s.min()),
            # Keep float precision: int() on (3+4)/2=3.5 silently truncates to 3.
            "median": round(float(s.median()), 1),
            "p95": round(float(s.quantile(0.95)), 1),
            "max": int(s.max()),
            "total": int(s.sum()),
        })
    out.write(_md(pd.DataFrame(rows), index=False) + "\n\n")

    out.write(_h(3, "Orphan checks (raw tables)"))
    out.write(
        "_The exported views inner-join through parents and so cannot contain "
        "orphans by construction. These counts query the underlying tables._\n\n"
    )
    orphans = (
        ("results -> athletes",
         "SELECT COUNT(*) FROM results r LEFT JOIN athletes a ON r.athlete_id=a.id WHERE a.id IS NULL"),
        ("results -> competitions",
         "SELECT COUNT(*) FROM results r LEFT JOIN competitions c ON r.competition_id=c.id WHERE c.id IS NULL"),
        ("round_results -> athletes",
         "SELECT COUNT(*) FROM round_results rr LEFT JOIN athletes a ON rr.athlete_id=a.id WHERE a.id IS NULL"),
        ("round_results -> competitions",
         "SELECT COUNT(*) FROM round_results rr LEFT JOIN competitions c ON rr.competition_id=c.id WHERE c.id IS NULL"),
        ("round_results -> category_rounds",
         "SELECT COUNT(*) FROM round_results rr LEFT JOIN category_rounds cr ON rr.category_round_id=cr.id WHERE cr.id IS NULL"),
        ("ascents -> athletes",
         "SELECT COUNT(*) FROM ascents a LEFT JOIN athletes at ON a.athlete_id=at.id WHERE at.id IS NULL"),
        ("ascents -> competitions",
         "SELECT COUNT(*) FROM ascents a LEFT JOIN competitions c ON a.competition_id=c.id WHERE c.id IS NULL"),
        ("ascents -> round_stages",
         "SELECT COUNT(*) FROM ascents a LEFT JOIN round_stages rs ON a.round_stage_id=rs.id WHERE rs.id IS NULL"),
        ("ascents -> routes",
         "SELECT COUNT(*) FROM ascents a LEFT JOIN routes rt ON a.route_id=rt.id WHERE rt.id IS NULL"),
        ("stage_results -> athletes",
         "SELECT COUNT(*) FROM stage_results sr LEFT JOIN athletes a ON sr.athlete_id=a.id WHERE a.id IS NULL"),
        ("stage_results -> competitions",
         "SELECT COUNT(*) FROM stage_results sr LEFT JOIN competitions c ON sr.competition_id=c.id WHERE c.id IS NULL"),
        ("stage_results -> round_stages",
         "SELECT COUNT(*) FROM stage_results sr LEFT JOIN round_stages rs ON sr.round_stage_id=rs.id WHERE rs.id IS NULL"),
        ("category_rounds -> competitions",
         "SELECT COUNT(*) FROM category_rounds cr LEFT JOIN competitions c ON cr.competition_id=c.id WHERE c.id IS NULL"),
        ("round_stages -> category_rounds",
         "SELECT COUNT(*) FROM round_stages rs LEFT JOIN category_rounds cr ON rs.category_round_id=cr.id WHERE cr.id IS NULL"),
        ("routes -> category_rounds",
         "SELECT COUNT(*) FROM routes rt LEFT JOIN category_rounds cr ON rt.category_round_id=cr.id WHERE cr.id IS NULL"),
        ("cup_rankings -> athletes",
         "SELECT COUNT(*) FROM cup_rankings cr LEFT JOIN athletes a ON cr.athlete_id=a.id WHERE a.id IS NULL"),
    )
    orphan_rows: list[dict[str, object]] = []
    for label, sql in orphans:
        df = execute_sql(sql, db_path=db_path)
        orphan_rows.append({"relation": label, "orphan_count": int(df.iat[0, 0])})
    out.write(_md(pd.DataFrame(orphan_rows), index=False) + "\n\n")


def _display_db_path(db_path: Path) -> str:
    """Render db_path relative to REPO_ROOT (with POSIX slashes) when possible,
    so the committed report doesn't leak per-developer absolute paths.
    Falls back to basename for paths outside the repo."""
    try:
        return str(db_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return db_path.name


def _read_schema_version(db_path: Path) -> int | None:
    """Read the actual schema version from the DB. The loader opens read-only,
    so `apply_schema` never runs - reporting the Python constant would lie
    when the on-disk warehouse is at an older version than the installed
    `wcl_data` package."""
    try:
        df = execute_sql("SELECT MAX(version) AS v FROM schema_version", db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("schema_version read failed: %s", exc)
        return None
    val = df.iat[0, 0]
    return int(val) if pd.notna(val) else None


def render_report(
    out_path: Path = DEFAULT_REPORT_PATH,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> Path:
    """Compute all profiling sections and write the markdown report.

    Returns the path written. Overwrites any existing file at `out_path`.
    """
    log.info("profiling warehouse at %s", db_path)
    buf = io.StringIO()
    buf.write("# Warehouse data profile\n\n")
    buf.write(
        "> _Auto-generated by `python -m wcl_analytics.profiling`. "
        "Re-run after any warehouse refresh. Do not hand-edit._\n\n"
    )
    schema_v = _read_schema_version(db_path)
    schema_display = f"v{schema_v}" if schema_v is not None else "(unknown - read failed)"
    buf.write(f"- **Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    buf.write(f"- **Source:** `{_display_db_path(db_path)}` (SQLite, read-only)\n")
    buf.write(f"- **Schema version:** {schema_display}\n")
    buf.write("- **Spec:** [`profiling-spec.md`](./profiling-spec.md)\n")

    # Cache competitions + athletes — both are reused by _climbing_extras, so
    # loading once avoids a second DB round-trip and a second leaked-handle
    # window.
    competitions_df: pd.DataFrame | None = None
    athletes_df: pd.DataFrame | None = None
    for name in VIEW_NAMES:
        log.info("profiling view: %s", name)
        if name == "ascents":
            df = load_view(name, db_path=db_path, limit=ASCENTS_SAMPLE_SIZE)
            _profile_view(name, df, buf, sampled=True)
        else:
            df = load_view(name, db_path=db_path)
            _profile_view(name, df, buf)
            if name == "competitions":
                competitions_df = df
            elif name == "athletes":
                athletes_df = df

    log.info("computing climbing-specific extras")
    assert competitions_df is not None and athletes_df is not None, (
        "competitions and athletes views must precede climbing_extras"
    )
    _climbing_extras(buf, db_path=db_path, competitions=competitions_df, athletes=athletes_df)

    log.info("computing cross-view consistency")
    _consistency(buf, db_path=db_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" overrides Path.write_text's default Windows LF→CRLF
    # translation so the committed artefact has stable line endings across
    # platforms (the repo has no .gitattributes to enforce this).
    out_path.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    log.info("wrote profile (%d bytes) to %s", out_path.stat().st_size, out_path)
    return out_path


def main() -> None:
    from wcl_data.logging_setup import configure, reconfigure_stdio_utf8
    reconfigure_stdio_utf8()
    configure()
    render_report()


if __name__ == "__main__":
    main()
