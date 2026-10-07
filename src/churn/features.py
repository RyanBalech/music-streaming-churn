"""Aggregate Parquet event logs without loading all events into a DataFrame.

DuckDB performs the event/session aggregations. Only one row per eligible user
is returned to Python. All recency features use the same explicit cutoff.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PAGES = (
    "NextSong",
    "Thumbs Up",
    "Thumbs Down",
    "Roll Advert",
    "Add Friend",
    "Add to Playlist",
    "Error",
    "Help",
    "Settings",
    "Upgrade",
    "Downgrade",
    "Logout",
)
CORE = (
    "n_events",
    "n_songs",
    "n_unique_songs",
    "n_unique_artists",
    "n_sessions",
    "active_days",
    "total_length",
    "avg_length",
    "events_per_day",
    "songs_per_day",
    "sessions_per_day",
    "n_errors",
    "errors_per_event",
    "errors_per_session",
    "gender_M",
    "days_since_reg",
    "activity_span_days",
    "time_since_last_event",
    "paid_ratio",
    "last_level_paid",
    "level_changes",
)
FEATURES = (
    CORE
    + tuple("page_" + page.replace(" ", "_") for page in PAGES)
    + (
        "thumbs_up_ratio",
        "help_rate",
        "advert_rate",
        "upgrade_count",
        "downgrade_count",
        "net_upgrade",
        "mean_sess_len",
        "max_sess_len",
        "min_sess_len",
        "sessions_with_error",
        "sessions_with_error_ratio",
        "weekend_event_ratio",
        "night_event_ratio",
        "n_events_3d",
        "n_songs_3d",
        "active_days_3d",
        "n_sessions_3d",
        "events_3d_ratio",
        "songs_3d_ratio",
        "sessions_3d_ratio",
        "last_is_song",
        "last_is_help",
        "last_is_upgrade",
        "last_is_downgrade",
        "last_is_logout",
    )
)
REQUIRED = {
    "userId",
    "ts",
    "registration",
    "page",
    "song",
    "artist",
    "sessionId",
    "length",
    "gender",
    "level",
    "status",
}


@dataclass
class Cohort:
    frame: pd.DataFrame
    metadata: dict


def utc_naive(value):
    value = pd.Timestamp(value)
    return (
        value.tz_localize("UTC").tz_localize(None)
        if value.tzinfo is None
        else value.tz_convert("UTC").tz_localize(None)
    )


def timestamp_sql(column):
    # Explicit epoch milliseconds for numeric input; ISO/timestamp input otherwise.
    return f"CASE WHEN try_cast({column} AS BIGINT) IS NOT NULL THEN epoch_ms(try_cast({column} AS BIGINT)) ELSE try_cast({column} AS TIMESTAMP) END"


def build_cohort(
    path,
    cutoff,
    horizon_days=10,
    *,
    label=True,
    exclude_cancelled=True,
    memory_limit="2GB",
    threads=2,
    temp_directory=None,
):
    """Build an at-risk cohort using observations <= cutoff and labels after it.

    A labeled cohort requires logs through cutoff + horizon. Earlier cancellations
    are excluded by default. The historical notebook used all observed users;
    exclude_cancelled=False is available for studying that eligibility difference.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cutoff = utc_naive(cutoff)
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    end = cutoff + pd.Timedelta(days=horizon_days)
    config = {"memory_limit": memory_limit, "threads": str(threads)}
    if temp_directory is not None:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        config["temp_directory"] = str(Path(temp_directory).resolve())
    with duckdb.connect(config=config) as con:
        # Reader relation accepts the path as data, never interpolated SQL.
        con.read_parquet(str(path)).create_view("raw")
        columns = {row[0] for row in con.execute("DESCRIBE raw").fetchall()}
        missing = REQUIRED - columns
        if missing:
            raise ValueError("Missing event columns: " + ", ".join(sorted(missing)))
        invalid = con.execute(
            "SELECT count(*) FROM raw WHERE userId IS NOT NULL AND trim(cast(userId AS VARCHAR)) <> '' AND NOT regexp_full_match(trim(cast(userId AS VARCHAR)), '[0-9]+')"
        ).fetchone()[0]
        if invalid:
            raise ValueError("Nonempty userId values must be nonnegative integers")
        con.execute(f"""CREATE TEMP VIEW events AS SELECT
            try_cast(userId AS BIGINT) AS id,
            {timestamp_sql("ts")} AS event_time,
            {timestamp_sql("registration")} AS registered,
            cast(page AS VARCHAR) AS page, cast(song AS VARCHAR) AS song,
            cast(artist AS VARCHAR) AS artist, cast(sessionId AS VARCHAR) AS session,
            try_cast(length AS DOUBLE) AS length, cast(gender AS VARCHAR) AS gender,
            cast(level AS VARCHAR) AS level,
            coalesce(try_cast(status AS INTEGER), -1) <> 200 AS is_error
            FROM raw WHERE userId IS NOT NULL AND trim(cast(userId AS VARCHAR)) <> ''""")
        invalid_time = con.execute(
            "SELECT count(*) FROM events WHERE event_time IS NULL OR id IS NULL"
        ).fetchone()[0]
        if invalid_time:
            raise ValueError("Invalid event timestamp or user identifier")
        count, anonymous, first, last = con.execute(
            "SELECT (SELECT count(*) FROM raw), (SELECT count(*) FROM raw) - count(*), min(event_time), max(event_time) FROM events"
        ).fetchone()
        if label and (last is None or pd.Timestamp(last) < end):
            raise ValueError("Logs do not cover the complete prediction horizon")
        con.execute(
            "CREATE TEMP TABLE parameters AS SELECT ?::TIMESTAMP AS cutoff, ?::TIMESTAMP AS future_end",
            [cutoff.to_pydatetime(), end.to_pydatetime()],
        )
        con.execute(
            "CREATE TEMP TABLE prior_cancel AS SELECT DISTINCT id FROM events, parameters WHERE event_time <= cutoff AND page = 'Cancellation Confirmation'"
        )
        eligible = "AND id NOT IN (SELECT id FROM prior_cancel)" if exclude_cancelled else ""
        con.execute(
            f"CREATE TEMP VIEW observation AS SELECT * FROM events, parameters WHERE event_time <= cutoff {eligible}"
        )
        last_order = "struct_pack(ts := event_time, action := coalesce(page, ''), tier := coalesce(level, ''))"
        page_sql = ",".join(
            f"count(*) FILTER (WHERE page = '{page}') AS page_{page.replace(' ', '_')}"
            for page in PAGES
        )
        con.execute(f"""CREATE TEMP TABLE users AS SELECT id,
            count(*) AS n_events, count(*) FILTER (WHERE page = 'NextSong') AS n_songs,
            count(DISTINCT song) AS n_unique_songs, count(DISTINCT artist) AS n_unique_artists,
            count(DISTINCT session) AS n_sessions, count(DISTINCT cast(event_time AS DATE)) AS active_days,
            coalesce(sum(length), 0) AS total_length, coalesce(avg(length), 0) AS avg_length,
            count(*) FILTER (WHERE is_error) AS n_errors,
            coalesce(arg_min(gender, {last_order}) = 'M', false)::INTEGER AS gender_M,
            floor(epoch(max(cutoff) - min(registered))/86400) AS days_since_reg,
            floor(epoch(max(event_time) - min(event_time))/86400) AS activity_span_days,
            floor(epoch(max(cutoff) - max(event_time))/86400) AS time_since_last_event,
            avg((coalesce(level, '') = 'paid')::INTEGER) AS paid_ratio,
            coalesce(arg_max(level, {last_order}) = 'paid', false)::INTEGER AS last_level_paid,
            (count(DISTINCT level) > 1)::INTEGER AS level_changes,
            {page_sql},
            avg((isodow(event_time) >= 6)::INTEGER) AS weekend_event_ratio,
            avg((hour(event_time) < 6)::INTEGER) AS night_event_ratio,
            count(*) FILTER (WHERE event_time >= cutoff - INTERVAL '3 days') AS n_events_3d,
            count(*) FILTER (WHERE event_time >= cutoff - INTERVAL '3 days' AND page = 'NextSong') AS n_songs_3d,
            count(DISTINCT cast(event_time AS DATE)) FILTER (WHERE event_time >= cutoff - INTERVAL '3 days') AS active_days_3d,
            count(DISTINCT session) FILTER (WHERE event_time >= cutoff - INTERVAL '3 days') AS n_sessions_3d,
            arg_max(page, {last_order}) AS last_page
            FROM observation GROUP BY id""")
        # Missing session IDs are excluded from session statistics, like pandas groupby.
        con.execute("""CREATE TEMP TABLE sessions AS WITH sizes AS (
            SELECT id, session, count(*) AS n, bool_or(is_error) AS error
            FROM observation WHERE session IS NOT NULL GROUP BY id, session)
            SELECT id, avg(n) AS mean_sess_len, max(n) AS max_sess_len, min(n) AS min_sess_len,
            sum(error::INTEGER) AS sessions_with_error FROM sizes GROUP BY id""")
        f = con.execute("SELECT * FROM users LEFT JOIN sessions USING(id) ORDER BY id").df()
        if f.empty:
            raise ValueError("No eligible users before cutoff")
        for numerator, denominator, output in (
            ("n_events", "active_days", "events_per_day"),
            ("n_songs", "active_days", "songs_per_day"),
            ("n_sessions", "active_days", "sessions_per_day"),
            ("n_errors", "n_events", "errors_per_event"),
            ("n_errors", "n_sessions", "errors_per_session"),
            ("page_Help", "n_events", "help_rate"),
            ("page_Roll_Advert", "n_events", "advert_rate"),
            ("sessions_with_error", "n_sessions", "sessions_with_error_ratio"),
            ("n_events_3d", "n_events", "events_3d_ratio"),
            ("n_songs_3d", "n_songs", "songs_3d_ratio"),
            ("n_sessions_3d", "n_sessions", "sessions_3d_ratio"),
        ):
            f[output] = f[numerator].fillna(0) / f[denominator].clip(lower=1)
        f["thumbs_up_ratio"] = f.page_Thumbs_Up / (f.page_Thumbs_Up + f.page_Thumbs_Down + 1)
        f["upgrade_count"], f["downgrade_count"] = f.page_Upgrade, f.page_Downgrade
        f["net_upgrade"] = f.upgrade_count - f.downgrade_count
        for name, page in (
            ("song", "NextSong"),
            ("help", "Help"),
            ("upgrade", "Upgrade"),
            ("downgrade", "Downgrade"),
            ("logout", "Logout"),
        ):
            f["last_is_" + name] = f.last_page.eq(page).astype(int)
        result = f[["id", *FEATURES]].replace([np.inf, -np.inf], np.nan).fillna(0)
        if label:
            churners = (
                con.execute(
                    "SELECT DISTINCT id FROM events, parameters WHERE event_time > cutoff AND event_time <= future_end AND page = 'Cancellation Confirmation'"
                )
                .df()
                .id
            )
            result["target"] = result.id.isin(churners).astype(int)
        metadata = {
            "input_rows": int(count),
            "anonymous_rows_removed": int(anonymous),
            "event_start": str(first),
            "event_end": str(last),
            "cutoff": str(cutoff),
            "horizon_end": str(end) if label else None,
            "eligible_users": len(result),
            "prior_cancel_users": int(
                con.execute("SELECT count(*) FROM prior_cancel").fetchone()[0]
            ),
            "excluded_prior_cancellations": bool(exclude_cancelled),
            "predictors": len(FEATURES),
            "event_clock": "ts; numeric values interpreted as epoch milliseconds",
            "latest_event_tie_break": "timestamp, page, level (lexicographic)",
            "memory_limit": memory_limit,
            "threads": threads,
        }
        return Cohort(result, metadata)
