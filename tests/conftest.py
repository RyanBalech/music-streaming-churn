import pandas as pd
import pytest


@pytest.fixture
def events():
    rows = []
    for user, date, page, session in [
        (1, "2020-01-01", "NextSong", 1),
        (1, "2020-01-09", "Help", 2),
        (2, "2020-01-03", "NextSong", 1),
        (2, "2020-01-08", "Logout", 1),
        (3, "2020-01-02", "NextSong", 1),
        (3, "2020-01-07", "Cancellation Confirmation", 2),
        (1, "2020-01-11", "Cancellation Confirmation", 3),
        (2, "2020-01-20", "NextSong", 3),
    ]:
        rows.append(
            {
                "userId": str(user),
                "ts": int(pd.Timestamp(date, tz="UTC").timestamp() * 1000),
                "registration": int(pd.Timestamp("2019-12-01", tz="UTC").timestamp() * 1000),
                "page": page,
                "sessionId": session,
                "song": "track" if page == "NextSong" else None,
                "artist": "artist" if page == "NextSong" else None,
                "length": 180.0 if page == "NextSong" else None,
                "gender": "M",
                "level": "paid",
                "status": 200,
            }
        )
    return pd.DataFrame(rows)
