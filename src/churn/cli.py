"""CLI for private-data evaluation and the synthetic engineering demo."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .demo import CUTOFF, make_demo
from .evaluation import (
    ModelConfig,
    environment,
    evaluate,
    file_sha256,
    fit_models,
    metrics,
    predictions,
    save_run,
    bootstrap_intervals,
)
from .features import build_cohort, utc_naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a synthetic smoke experiment; no competition data")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--users", type=int, default=300)
    demo.add_argument("--seed", type=int, default=2026)
    for command in ("evaluate", "forward"):
        p = sub.add_parser(command)
        p.add_argument("--events", type=Path, required=True)
        p.add_argument("--cutoff", required=True, help="UTC training observation cutoff")
        p.add_argument("--horizon-days", type=int, default=10)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--memory-limit", default="2GB")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--bootstrap", type=int, default=1000)
        if command == "forward":
            p.add_argument("--evaluation-cutoff", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        source = make_demo(args.output / "synthetic_events.parquet", args.users, args.seed)
        cutoff, horizon, threads, memory, bootstrap = CUTOFF, 10, 2, "512MB", 100
        config = ModelConfig(gb_estimators=30, rf_estimators=40)
        scope = "synthetic software smoke experiment; not evidence of real-data performance"
    else:
        source = args.events
        cutoff, horizon, threads, memory, bootstrap = (
            args.cutoff,
            args.horizon_days,
            args.threads,
            args.memory_limit,
            args.bootstrap,
        )
        config = ModelConfig(threads=threads)
        scope = "private event logs; forward cancellation labels"
    if args.command == "forward":
        evaluation_cutoff = utc_naive(args.evaluation_cutoff)
        training_label_end = utc_naive(cutoff) + pd.Timedelta(days=horizon)
        if evaluation_cutoff < training_label_end:
            raise ValueError("Training labels must finish before the evaluation cutoff")
    cohort = build_cohort(
        source,
        cutoff,
        horizon,
        memory_limit=memory,
        threads=threads,
        temp_directory=args.output / "duckdb_tmp",
    )
    report, scores, models = evaluate(cohort.frame, config, bootstrap)
    report.update(
        scope=scope,
        cohort=cohort.metadata,
        input_sha256=file_sha256(source),
        environment=environment(),
    )
    if args.command == "forward":
        later = build_cohort(
            source,
            evaluation_cutoff,
            horizon,
            memory_limit=memory,
            threads=threads,
            temp_directory=args.output / "duckdb_tmp",
        )
        # Refit on the complete earlier cohort; all its labels are available by the later cutoff.
        models = fit_models(cohort.frame, config)
        pf = predictions(models, later.frame, config)
        report["forward"] = {
            "protocol": "earlier labeled cohort to later observation cutoff; fixed validation thresholds",
            "cohort": later.metadata,
            "models": {},
        }
        scores = later.frame[["id", "target"]].copy()
        for name, probability in pf.items():
            threshold = report["models"][name]["validation"]["threshold"]
            report["forward"]["models"][name] = {
                "metrics": metrics(later.frame.target, probability, threshold),
                "bootstrap": bootstrap_intervals(
                    later.frame.target, probability, bootstrap, config.seed
                ),
            }
            scores[name] = probability
    save_run(args.output, report, scores, models)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scope": scope,
                "users": len(cohort.frame),
                "ensemble_test_auc": report["models"]["ensemble"]["test"]["roc_auc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
