"""World Climbing Lab — analytics layer (`wcl_analytics`).

Layer 1 atop the `wcl_data` ingestion warehouse. The first module is data
profiling (`profiling.py`); future modules will share `loaders.py` for
read-only DataFrame access to the warehouse views.
"""
from __future__ import annotations

from .loaders import DEFAULT_DB_PATH, execute_sql, load_view

__version__ = "0.1.0"
__all__ = ["DEFAULT_DB_PATH", "execute_sql", "load_view"]
