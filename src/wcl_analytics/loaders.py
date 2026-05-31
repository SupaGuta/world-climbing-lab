"""Read-only loaders for the warehouse views.

Materializes the denormalized SELECTs defined in `wcl_data.exporter.VIEWS`
against the SQLite warehouse and returns pandas DataFrames. Read-only by
construction: opens the DB via the `file:...?mode=ro` URI form. Shared by
every `wcl_analytics` module so the view-to-DataFrame mapping has exactly
one definition.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from urllib.request import pathname2url

import pandas as pd

from wcl_data.config import REPO_ROOT
from wcl_data.exporter import VIEWS, VIEW_NAMES

log = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = REPO_ROOT / "data" / "wcl.sqlite"

# Word-bounded `ORDER BY` matcher. Used by `_strip_trailing_order_by` to
# locate occurrences when deciding whether the LIMIT-wrap is safe.
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _strip_trailing_order_by(sql: str) -> str:
    """Remove a trailing `ORDER BY ...` clause if exactly one is present.

    Stripping the sort is a perf win on `ascents` (~880k rows): the LIMIT
    wrapper would otherwise force SQLite to sort the entire table before
    applying LIMIT. When the SQL contains more than one `ORDER BY` (CTE,
    window function, nested subquery), we cannot safely decide which one
    is the trailing/outermost without parsing — return the SQL untouched.
    """
    matches = list(_ORDER_BY_PATTERN.finditer(sql))
    if len(matches) != 1:
        return sql
    return sql[: matches[0].start()].rstrip()


@contextmanager
def _open_ro(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the warehouse read-only and guarantee the connection is closed.

    `sqlite3.Connection`'s own context manager only commits/rolls back; it
    does NOT close. Each call would otherwise leak a file handle, which on
    Windows blocks subsequent writers on `data/wcl.sqlite`. URL-encoding
    the path via `pathname2url` is required because raw spaces or `?` in a
    Windows path would otherwise truncate the `file:` URI.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"warehouse not found at {db_path}")
    uri = f"file:{pathname2url(str(db_path))}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        yield conn


def load_view(
    name: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    limit: int | None = None,
    parse_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load one warehouse view as a DataFrame.

    `name` must be one of `wcl_data.exporter.VIEW_NAMES`. `limit` samples the
    first N rows (the inner `ORDER BY` is stripped where safe, so the sample
    is unordered but cheap on large views like `ascents`). `parse_dates` is
    forwarded to `pd.read_sql_query`.
    """
    if name not in VIEWS:
        raise ValueError(f"unknown view {name!r}; expected one of {VIEW_NAMES}")
    if limit is not None and limit < 1:
        # SQLite treats LIMIT -1 as "no limit" — guard against the user
        # accidentally loading the whole 880k-row ascents view via a -1
        # sentinel. limit=0 is also disallowed (use a separate code path).
        raise ValueError(f"limit must be >= 1, got {limit}")
    sql = VIEWS[name].strip()
    if limit is not None:
        sql = f"SELECT * FROM ({_strip_trailing_order_by(sql)}) LIMIT {int(limit)}"
    with _open_ro(db_path) as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            parse_dates=list(parse_dates) if parse_dates else None,
        )
    log.debug("loaded view %s: %d rows x %d cols", name, len(df), df.shape[1])
    return df


def execute_sql(
    sql: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    params: Sequence[object] = (),
) -> pd.DataFrame:
    """Run an arbitrary read-only SELECT and return the result as a DataFrame.

    Escape hatch for cross-view aggregates (orphan checks, per-discipline
    `ascents` fill-rates, join cardinality) where materializing whole views
    would be wasteful.
    """
    with _open_ro(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=tuple(params))
