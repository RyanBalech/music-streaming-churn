import numpy as np
import pandas as pd
import pytest
from churn.features import FEATURES, build_cohort

CUTOFF = "2020-01-10"


def build(events, tmp_path, **kwargs):
    path = tmp_path / "events.parquet"
    events.to_parquet(path, index=False)
    return build_cohort(path, CUTOFF, **kwargs)


def test_future_window_and_at_risk_cohort(events, tmp_path):
    cohort = build(events, tmp_path)
    assert cohort.frame.id.tolist() == [1, 2]
    assert cohort.frame.target.tolist() == [1, 0]
    assert cohort.metadata["prior_cancel_users"] == 1
    assert cohort.frame.set_index("id").loc[1, "n_events"] == 2
    assert cohort.frame.set_index("id").loc[1, "days_since_reg"] == 40


def test_future_behavior_cannot_change_features(events, tmp_path):
    before = build(events, tmp_path).frame
    events.loc[
        events.ts > pd.Timestamp(CUTOFF).timestamp() * 1000, ["page", "length", "gender", "level"]
    ] = ["Error", 999999.0, "F", "free"]
    after = build(events, tmp_path).frame
    pd.testing.assert_frame_equal(before[["id", *FEATURES]], after[["id", *FEATURES]])
    assert before.target.sum() == 1 and after.target.sum() == 0


def test_row_order_does_not_change_last_action(events, tmp_path):
    first = build(events, tmp_path).frame
    second = build(events.sample(frac=1, random_state=9), tmp_path).frame
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("id").loc[1, "last_is_help"] == 1


def test_fixed_schema_when_pages_are_missing(events, tmp_path):
    events.loc[events.page == "Help", "page"] = "NextSong"
    result = build(events, tmp_path).frame
    assert len(FEATURES) == 58
    assert result.columns.tolist() == ["id", *FEATURES, "target"]
    assert result.page_Help.sum() == 0
    assert np.isfinite(result[list(FEATURES)]).all().all()


def test_timestamp_and_epoch_ms_inputs_agree(events, tmp_path):
    first = build(events, tmp_path).frame
    for column in ("ts", "registration"):
        events[column] = pd.to_datetime(events[column], unit="ms")
    second = build(events, tmp_path).frame
    pd.testing.assert_frame_equal(first, second)


def test_horizon_must_be_observed(events, tmp_path):
    events = events.iloc[:-1]
    with pytest.raises(ValueError, match="complete prediction horizon"):
        build(events, tmp_path)
    assert len(build(events, tmp_path, label=False).frame) == 2


def test_cutoff_is_observed_not_a_future_label(events, tmp_path):
    row = events.iloc[0].copy()
    row["ts"] = int(pd.Timestamp(CUTOFF).timestamp() * 1000)
    row["page"] = "Cancellation Confirmation"
    result = build(
        pd.concat([events, row.to_frame().T], ignore_index=True).astype(
            {"ts": "int64", "registration": "int64"}
        ),
        tmp_path,
    )
    assert result.frame.id.tolist() == [2]


def test_cancellations_after_horizon_are_not_labels(events, tmp_path):
    events.loc[events.page == "Cancellation Confirmation", "ts"] = int(
        pd.Timestamp("2020-01-21").timestamp() * 1000
    )
    assert build(events, tmp_path).frame.target.sum() == 0


def test_anonymous_events_are_counted_and_removed(events, tmp_path):
    row = events.iloc[0].copy()
    row["userId"] = ""
    result = build(
        pd.concat([events, row.to_frame().T], ignore_index=True).astype(
            {"ts": "int64", "registration": "int64"}
        ),
        tmp_path,
    )
    assert result.metadata["anonymous_rows_removed"] == 1
    assert result.frame.id.tolist() == [1, 2]


@pytest.mark.parametrize("column,value", [("userId", "oops"), ("ts", "bad")])
def test_malformed_identifiers_and_timestamps_fail(events, tmp_path, column, value):
    events[column] = events[column].astype(str)
    events.loc[0, column] = value
    with pytest.raises(ValueError):
        build(events, tmp_path)


def test_missing_columns_fail(events, tmp_path):
    with pytest.raises(ValueError, match="Missing event columns"):
        build(events.drop(columns="page"), tmp_path)
