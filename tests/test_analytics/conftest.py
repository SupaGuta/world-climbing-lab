"""Fixtures for wcl_analytics tests.

`loaders.load_view` opens the warehouse via the `file:...?mode=ro` URI, which
cannot point at a `:memory:` DB. Tests therefore use a tmp-path file seeded
through the real `wcl_data.db.schema.open_db` path (FK + WAL enforced, same as
production) so the read-only loader is exercised end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wcl_data.db.schema import open_db


_SEED_SQL = """
INSERT INTO disciplines (id, name) VALUES (1, 'lead'), (2, 'boulder'), (3, 'speed');
INSERT INTO categories (id, name, gender) VALUES (1, 'Men', 0), (2, 'Women', 1);

INSERT INTO seasons (id, ifsc_id, year, last_fetched_at)
VALUES (1, 2024, 2024, '2024-06-05 12:00:00');

INSERT INTO leagues (id, name) VALUES (1, 'World Cup');

INSERT INTO events (id, ifsc_id, season_id, league_id, name, city, country, country_iso3,
                    date_start, date_end, is_paraclimbing, last_fetched_at)
VALUES (1, 1001, 1, 1, 'Test Event', 'Innsbruck', 'Austria', 'AUT',
        '2024-06-01', '2024-06-03', 0, '2024-06-05 12:00:00');

INSERT INTO competitions (id, event_id, discipline_id, category_id, ifsc_id) VALUES
  (1, 1, 1, 1, 1),
  (2, 1, 2, 1, 2),
  (3, 1, 3, 1, 3);

INSERT INTO athletes (id, ifsc_id, firstname, lastname, gender, height, arm_span,
                      birthday, country, country_iso3, speed_pb_time, speed_pb_date) VALUES
  (1, 101, 'Adam',  'Ondra',    0, 185, 188, '1993-02-05', 'Czechia',  'CZE', NULL,   NULL),
  (2, 102, 'Janja', 'Garnbret', 1, 164, 164, '1999-03-12', 'Slovenia', 'SVN', NULL,   NULL),
  (3, 103, 'Veddriq','Leonardo',0, 168, NULL,'1997-07-11', 'Indonesia','IDN', '4.90', '2023-08-12');

INSERT INTO cup_rankings (id, athlete_id, cup_ifsc_id, cup_name, season, discipline, d_cat_id, rank) VALUES
  (1, 1, 2000, 'World Cup 2024', '2024', 'lead',    11, 1),
  (2, 2, 2000, 'World Cup 2024', '2024', 'lead',    12, 2),
  (3, 3, 2000, 'World Cup 2024', '2024', 'speed',   NULL, 1);

INSERT INTO category_rounds (id, ifsc_id, competition_id, kind, name, category, format, league_round_id) VALUES
  (1, 3001, 1, 'qualification', 'Qualification', 'Men', 'lead',    1),
  (2, 3002, 2, 'qualification', 'Qualification', 'Men', 'boulder', 1),
  (3, 3003, 3, 'qualification', 'Qualification', 'Men', 'speed',   1);

INSERT INTO round_stages (id, category_round_id, seq, name, kind) VALUES
  (1, 1, 0, 'Stage 1', 'lead'),
  (2, 2, 0, 'Stage 1', 'boulder'),
  (3, 3, 0, 'Stage 1', 'speed');

INSERT INTO routes (id, ifsc_id, category_round_id, name) VALUES
  (1, 4001, 1, 'Route 1'),
  (2, 4002, 2, 'Boulder 1'),
  (3, 4003, 3, 'Speed Wall');

INSERT INTO results (id, competition_id, athlete_id, rank) VALUES
  (1, 1, 1, 1),
  (2, 1, 2, 2),
  (3, 2, 1, 3);

INSERT INTO round_results (id, competition_id, category_round_id, athlete_id, rank, score, starting_group) VALUES
  (1, 1, 1, 1, 1, 'TOP', 'A'),
  (2, 1, 1, 2, 2, '50+', 'B'),
  (3, 2, 2, 1, 3, '3T 4Z', NULL);

-- Spread ascents across all three disciplines so the per-discipline score-fill
-- aggregate has a row per discipline.
INSERT INTO ascents (id, competition_id, round_stage_id, route_id, athlete_id, rank, score,
                     top, plus, top_tries, time_ms, zone, zone_tries, dnf, dns, points) VALUES
  (1, 1, 1, 1, 1, 1, 'TOP', 1, 0, 1,    NULL, NULL, NULL, 0, 0, NULL),
  (2, 1, 1, 1, 2, 2, '50+', 0, 1, 2,    NULL, NULL, NULL, 0, 0, NULL),
  (3, 2, 2, 2, 1, 1, NULL,  NULL, NULL, 3,    NULL, 1,    2, 0, 0, 25.0),
  (4, 2, 2, 2, 2, 2, NULL,  NULL, NULL, NULL, NULL, 0,    0, 1, 0, 0.0),
  (5, 3, 3, 3, 1, 1, '5.43',NULL, NULL, NULL, 5430, NULL, NULL, 0, 0, NULL),
  (6, 3, 3, 3, 2, 2, '5.89',NULL, NULL, NULL, 5890, NULL, NULL, 0, 0, NULL);
"""


@pytest.fixture
def warehouse_db(tmp_path: Path) -> Path:
    """A tmp-path SQLite warehouse seeded with one event across 3 disciplines.

    Returns the path; tests open it read-only through `loaders`.
    """
    path = tmp_path / "wcl.sqlite"
    conn = open_db(path)
    try:
        conn.executescript(_SEED_SQL)
        conn.commit()
    finally:
        conn.close()
    return path
