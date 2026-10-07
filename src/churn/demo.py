"""Deterministic fabricated events for software checks, not research results."""

from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2018-11-10", tz="UTC")


def make_demo(path, users=300, seed=2026):
    if users < 60:
        raise ValueError("Demo needs at least 60 users")
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(users):
        # Latent risk changes engagement and future cancellation independently.
        risk = float(rng.beta(2, 4))
        churn = bool(rng.random() < 0.08 + 0.45 * risk)
        n = int(rng.integers(25, 75))
        registered = int(
            (CUTOFF - pd.Timedelta(days=int(rng.integers(60, 160)))).timestamp() * 1000
        )
        for index in range(n):
            age = float(rng.uniform(0, 38)) + risk * 2
            ts = CUTOFF - pd.Timedelta(days=age)
            page = rng.choice(
                ["NextSong", "Thumbs Up", "Thumbs Down", "Roll Advert", "Help", "Logout"],
                p=[0.65, 0.1, 0.05, 0.1, 0.05, 0.05],
            )
            rows.append(
                dict(
                    userId=str(user),
                    ts=int(ts.timestamp() * 1000),
                    registration=registered,
                    page=page,
                    song=f"song_{rng.integers(100)}" if page == "NextSong" else None,
                    artist=f"artist_{rng.integers(20)}" if page == "NextSong" else None,
                    sessionId=index // 8,
                    length=float(rng.uniform(100, 300)) if page == "NextSong" else None,
                    gender="M" if user % 2 else "F",
                    level="paid" if user % 3 else "free",
                    status=200,
                )
            )
        row = rows[-1].copy()
        row.update(
            ts=int((CUTOFF + pd.Timedelta(days=10)).timestamp() * 1000),
            page="Cancellation Confirmation" if churn else "NextSong",
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
