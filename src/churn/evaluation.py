"""User-disjoint evaluation with validation-only threshold selection."""

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    gb_estimators: int = 600
    rf_estimators: int = 500
    threads: int = 2
    gb_weight: float = 0.7

    def __post_init__(self):
        if (
            not 0 <= self.gb_weight <= 1
            or min(self.gb_estimators, self.rf_estimators, self.threads) <= 0
        ):
            raise ValueError("Invalid ensemble weight or model size")


def split_users(frame, seed=42):
    if frame.id.duplicated().any():
        raise ValueError("Expected one row per user")
    if (
        not frame.target.isin([0, 1]).all()
        or frame.target.value_counts().min() < 10
        or frame.target.nunique() != 2
    ):
        raise ValueError("At least ten users in each binary class are needed for the split")
    train, heldout = train_test_split(
        frame, test_size=0.4, stratify=frame.target, random_state=seed
    )
    valid, test = train_test_split(
        heldout, test_size=0.5, stratify=heldout.target, random_state=seed + 1
    )
    return train, valid, test


def choose_threshold(y, probability):
    """Maximize F1 on validation labels; prefer the higher threshold on a tie."""
    y, probability = validate_predictions(y, probability)
    if np.unique(y).size != 2:
        raise ValueError("Threshold selection requires both classes")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def validate_predictions(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.size == 0 or y.shape != probability.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned one-dimensional binary labels and probabilities")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    return y, probability


def metrics(y, probability, threshold=0.5):
    y, probability = validate_predictions(y, probability)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    pred = probability >= threshold
    ranked = np.argsort(-probability, kind="stable")
    top = ranked[: max(1, int(np.ceil(0.1 * len(y))))]
    prevalence = float(y.mean())
    return {
        "users": len(y),
        "churn_rate": prevalence,
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, probability)) if y.sum() else None,
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "top_10_percent_precision": float(y[top].mean()),
        "top_10_percent_recall": float(y[top].sum() / y.sum()) if y.sum() else None,
        "top_10_percent_lift": float(y[top].mean() / prevalence) if prevalence else None,
    }


def bootstrap_intervals(y, probability, repetitions=1000, seed=2026):
    """Stratified user bootstrap conditional on observed prevalence and fixed scores."""
    y, probability = validate_predictions(y, probability)
    if repetitions < 1 or np.unique(y).size != 2:
        raise ValueError("Bootstrap needs both classes and positive repetitions")
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    auc, ap = [], []
    for _ in range(repetitions):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        auc.append(roc_auc_score(y[ids], probability[ids]))
        ap.append(average_precision_score(y[ids], probability[ids]))
    return {
        "method": "stratified user bootstrap; fixed predictions, conditional on observed prevalence",
        "repetitions": repetitions,
        "seed": seed,
        "roc_auc_95": np.quantile(auc, [0.025, 0.975]).tolist(),
        "average_precision_95": np.quantile(ap, [0.025, 0.975]).tolist(),
    }


def fit_models(frame, config):
    x, y = frame[list(FEATURES)], frame.target
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.seed),
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=config.gb_estimators,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            random_state=config.seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=config.threads,
            random_state=config.seed,
        ),
    }
    for model in models.values():
        model.fit(x, y)
    return models


def predictions(models, frame, config):
    values = {
        name: model.predict_proba(frame[list(FEATURES)])[:, 1] for name, model in models.items()
    }
    values["ensemble"] = (
        config.gb_weight * values["gradient_boosting"]
        + (1 - config.gb_weight) * values["random_forest"]
    )
    return values


def evaluate(frame, config=ModelConfig(), bootstrap=1000):
    train, valid, test = split_users(frame, config.seed)
    models = fit_models(train, config)
    pv, pt = predictions(models, valid, config), predictions(models, test, config)
    pv["prior"] = np.full(len(valid), train.target.mean())
    pt["prior"] = np.full(len(test), train.target.mean())
    reports = {}
    for name in pv:
        threshold = 0.5 if name == "prior" else choose_threshold(valid.target, pv[name])
        reports[name] = {
            "validation": metrics(valid.target, pv[name], threshold),
            "test": metrics(test.target, pt[name], threshold),
            "test_bootstrap": bootstrap_intervals(test.target, pt[name], bootstrap, config.seed),
        }
    # The ranking budget is fixed; the ensemble weight is inherited, not selected on test.
    output = test[["id", "target"]].copy()
    for name, probability in pt.items():
        output[name] = probability
    manifest = {
        part: {
            "users": len(data),
            "churners": int(data.target.sum()),
            "user_ids_sha256": hashlib.sha256(
                np.sort(data.id.to_numpy()).astype("<i8").tobytes()
            ).hexdigest(),
        }
        for part, data in (("train", train), ("validation", valid), ("test", test))
    }
    return (
        {
            "protocol": "60/20/20 stratified, disjoint users at one fixed cutoff; not a chronological model backtest",
            "config": asdict(config),
            "splits": manifest,
            "models": reports,
        },
        output,
        models,
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    packages = ["duckdb", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
    return {
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
    }


def save_run(directory, report, scores, models):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scores.to_csv(directory / "heldout_predictions.csv", index=False)
    joblib.dump(
        {"features": list(FEATURES), "models": models, "config": report["config"]},
        directory / "models.joblib",
    )
