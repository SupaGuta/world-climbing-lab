"""Tests for `wcl_analytics.loaders`."""
from __future__ import annotations

from pathlib import Path

import pytest

from wcl_analytics.loaders import (
    _strip_trailing_order_by,
    execute_sql,
    load_view,
)


def test_load_seasons(warehouse_db: Path) -> None:
    df = load_view("seasons", db_path=warehouse_db)
    assert list(df.columns) == ["season_ifsc_id", "year", "last_fetched_at"]
    assert len(df) == 1
    assert int(df.loc[0, "year"]) == 2024


def test_load_ascents_limit(warehouse_db: Path) -> None:
    full = load_view("ascents", db_path=warehouse_db)
    sampled = load_view("ascents", db_path=warehouse_db, limit=3)
    assert len(full) == 6
    assert len(sampled) == 3
    assert list(sampled.columns) == list(full.columns)


def test_load_view_unknown_name_raises(warehouse_db: Path) -> None:
    with pytest.raises(ValueError, match="unknown view"):
        load_view("not_a_real_view", db_path=warehouse_db)


def test_load_view_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_view("seasons", db_path=tmp_path / "missing.sqlite")


def test_load_view_rejects_zero_limit(warehouse_db: Path) -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        load_view("ascents", db_path=warehouse_db, limit=0)


def test_load_view_rejects_negative_limit(warehouse_db: Path) -> None:
    # SQLite treats LIMIT -1 as "no limit", which would silently load the
    # whole 880k-row ascents view. The loader must reject this up front.
    with pytest.raises(ValueError, match="limit must be >= 1"):
        load_view("ascents", db_path=warehouse_db, limit=-1)


def test_strip_trailing_order_by_single_clause() -> None:
    sql = "SELECT a, b FROM t WHERE x = 1 ORDER BY a DESC, b"
    stripped = _strip_trailing_order_by(sql)
    assert "ORDER BY" not in stripped.upper()
    assert "WHERE x = 1" in stripped


def test_strip_trailing_order_by_noop_when_absent() -> None:
    sql = "SELECT 1 AS x"
    assert _strip_trailing_order_by(sql) == sql


def test_strip_trailing_order_by_leaves_multi_clause_untouched() -> None:
    # A future view with a CTE/window ORDER BY must not have its inner sort
    # corrupted by a naive trailing-strip — we'd rather pay for the outer
    # sort than emit broken SQL.
    sql = (
        "WITH ranked AS (SELECT *, ROW_NUMBER() OVER (ORDER BY x) AS r FROM t)"
        " SELECT * FROM ranked ORDER BY r"
    )
    assert _strip_trailing_order_by(sql) == sql


def test_load_view_path_with_space_works(tmp_path: Path, warehouse_db: Path) -> None:
    # `pathname2url` is required because raw `file:C:\Users\Jane Doe\...`
    # would otherwise truncate the URI at the first unescaped space.
    spaced_dir = tmp_path / "with space"
    spaced_dir.mkdir()
    spaced_path = spaced_dir / "wcl.sqlite"
    spaced_path.write_bytes(warehouse_db.read_bytes())
    df = load_view("seasons", db_path=spaced_path)
    assert len(df) == 1


def test_execute_sql_orphan_query_returns_zero(warehouse_db: Path) -> None:
    df = execute_sql(
        "SELECT COUNT(*) AS n FROM results r "
        "LEFT JOIN athletes a ON r.athlete_id = a.id WHERE a.id IS NULL",
        db_path=warehouse_db,
    )
    assert int(df.iat[0, 0]) == 0


def test_execute_sql_returns_dataframe(warehouse_db: Path) -> None:
    df = execute_sql("SELECT COUNT(*) AS n FROM ascents", db_path=warehouse_db)
    assert int(df.iat[0, 0]) == 6
