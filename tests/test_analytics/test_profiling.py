"""Tests for `wcl_analytics.profiling`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wcl_analytics import profiling


def test_dup_on_key_collapses_nulls() -> None:
    df = pd.DataFrame({"a": [1, 1, None, None], "b": [None, None, 2, 2]})
    assert profiling._dup_on_key(df, ("a", "b")) == 2


def test_dup_on_key_raises_when_column_missing() -> None:
    # Returning a sentinel like -1 would silently render as a "real" negative
    # duplicate count in the report. Raise instead so NATURAL_KEYS drift is
    # loud.
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError, match="natural-key columns"):
        profiling._dup_on_key(df, ("a", "missing"))


def test_md_escapes_pipes_in_cells() -> None:
    df = pd.DataFrame({"x": ["a|b", "c"]})
    out = profiling._md(df, index=False)
    assert "a\\|b" in out
    # Original pipe is escaped; tabulate's own structural pipes (column
    # separators) remain. The cell now contains `a\|b`, not `a|b`.
    assert "| a\\|b" in out or "a\\|b " in out


def test_md_escapes_pipes_in_index() -> None:
    df = pd.DataFrame({"count": [3]}, index=pd.Index(["bad|name"], name="cat"))
    out = profiling._md(df)
    assert "bad\\|name" in out


def test_natural_keys_in_sync_with_view_names() -> None:
    from wcl_data.exporter import VIEW_NAMES
    assert set(profiling.NATURAL_KEYS) == set(VIEW_NAMES)


def test_render_report_writes_all_sections(warehouse_db: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "profile.md"
    profiling.render_report(out_path, db_path=warehouse_db)
    body = out_path.read_text(encoding="utf-8")

    # Header
    assert body.startswith("# Warehouse data profile")
    assert "**Schema version:** v6" in body  # read from DB, not from constant
    # warehouse_db lives outside REPO_ROOT, so the display path falls back to
    # the basename — no absolute path / username should leak.
    assert "**Source:** `wcl.sqlite`" in body
    assert "Guta" not in body

    # All 9 view sections present (ascents marked sampled)
    for view in (
        "seasons", "leagues", "events", "competitions", "athletes",
        "cup_rankings", "results", "round_results",
    ):
        assert f"## `{view}`" in body, f"missing view section: {view}"
    assert "## `ascents` - sampled (LIMIT" in body  # loose match — seed size irrelevant

    # Each view has standard sub-sections
    assert body.count("### Per-column summary") == 9
    assert body.count("### Duplicates") == 9

    # Sampled view (ascents) marks its dup counts as on-sample
    assert "on sample only" in body

    # Climbing-specific extras
    assert "## Climbing-specific extras" in body
    assert "### Disciplines x categories x genders" in body
    assert "### `athletes` fill rates" in body
    assert "### `ascents` per-discipline score-column fill rates" in body

    # Cross-view consistency — must include stage_results orphans now
    assert "## Cross-view consistency" in body
    assert "### Join cardinality" in body
    assert "### Orphan checks" in body
    for relation in (
        "results -> athletes", "ascents -> routes",
        "category_rounds -> competitions", "cup_rankings -> athletes",
        "stage_results -> athletes", "stage_results -> competitions",
        "stage_results -> round_stages",
    ):
        assert relation in body, f"missing orphan check: {relation}"


def test_render_report_uses_lf_line_endings(warehouse_db: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "profile.md"
    profiling.render_report(out_path, db_path=warehouse_db)
    raw = out_path.read_bytes()
    assert b"\r\n" not in raw, "report must use LF line endings, not CRLF"


def test_render_report_orphans_all_zero_on_fk_warehouse(warehouse_db: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "profile.md"
    profiling.render_report(out_path, db_path=warehouse_db)
    body = out_path.read_text(encoding="utf-8")

    orphan_section = body.split("### Orphan checks")[1].split("\n## ")[0]
    data_rows = [
        line for line in orphan_section.splitlines()
        if line.startswith("|") and "->" in line
    ]
    assert data_rows, "expected at least one orphan-check row"
    for row in data_rows:
        count_cell = [c.strip() for c in row.split("|") if c.strip()][-1]
        assert count_cell == "0", f"non-zero orphans in row: {row}"


def test_competitions_semantic_dup_check_renders(warehouse_db: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "profile.md"
    profiling.render_report(out_path, db_path=warehouse_db)
    body = out_path.read_text(encoding="utf-8")
    assert "Duplicates on semantic key" in body


def test_competitions_natural_key_finds_no_dupes_on_clean_seed(warehouse_db: Path) -> None:
    from wcl_analytics.loaders import load_view
    df = load_view("competitions", db_path=warehouse_db)
    assert profiling._dup_on_key(df, profiling.NATURAL_KEYS["competitions"]) == 0


def test_per_discipline_fill_rates_use_left_join(warehouse_db: Path, tmp_path: Path) -> None:
    # The 'full table' label demands LEFT JOIN coverage so orphan ascents
    # surface as a <orphan> bucket instead of being silently excluded.
    out_path = tmp_path / "profile.md"
    profiling.render_report(out_path, db_path=warehouse_db)
    body = out_path.read_text(encoding="utf-8")
    extras = body.split("### `ascents` per-discipline score-column fill rates")[1]
    extras = extras.split("\n## ")[0]
    # All three real disciplines from the seed should appear.
    for d in ("lead", "boulder", "speed"):
        assert d in extras, f"missing discipline row in extras: {d}"
