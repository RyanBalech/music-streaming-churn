# Experimental protocol

## Historical evidence

The archived notebook is the original course workflow with local paths replaced by `CHURN_DATA_DIR` (default `../data`) and execution outputs removed. The aggregate ROC plot was extracted from its saved output. The manifest preserves the supplied notebook's SHA-256 and exact recorded scores; it does not represent a new execution.

The original workflow used a stratified 80/20 split of users at a single observation cutoff. It selected a competition submission threshold using a positive-rate quantile of 42.4%. That submission setting was leaderboard-oriented and is not a defensible general deployment rule, particularly when observed training prevalence was approximately 3.5%. A saved leaderboard-score artifact was not supplied; no leaderboard accuracy is reported here.

The original feature matrix contained 59 columns **including user ID**, hence 58 predictors. Course rank (1/40) is author-provided. The supplied presentation names another team and describes different experiments, so its model results are not attributed to this project.

## Revised evaluation

1. Construct the at-risk cohort using only pre-cutoff behavior, excluding earlier cancellations.
2. Allocate disjoint users to train/validation/test (60/20/20), stratified by target, seed 42. Fit preprocessing and models using training users only.
3. Select each model's decision threshold by validation F1. The constant prevalence baseline uses threshold 0.5. Keep the original 0.7/0.3 ensemble weight fixed.
4. Evaluate the untouched test users once. Report probability, ranking and threshold metrics separately; accuracy alone is insufficient for an imbalanced target.
5. Bootstrap fixed predictions within each class. Percentile intervals are conditional on the observed class balance and do not include model-training variability or temporal distribution shift.

Logistic regression uses median imputation, scaling and balanced class weights. Gradient boosting uses 600 trees, learning rate 0.04, depth 3 and subsampling 0.9. Random forest uses 500 trees, minimum split size 10 and minimum leaf size 5. The synthetic demo uses smaller ensembles for a fast engineering check.

Forward mode first produces the earlier held-out-user report, then refits on all earlier labeled users. It scores the later cohort with the previously selected thresholds. `models.joblib` and `heldout_predictions.csv` then correspond to this later evaluation. There is no later-window threshold tuning. The forward report covers the logistic, tree and ensemble models; the earlier report additionally includes the prevalence baseline.

## Interpretation and next experiment

The new implementation is validated with unit tests and fabricated event logs. A real-data rerun needs the original Parquet files. Run both the historical notebook and revised pipeline on the same input, then isolate eligibility, event ordering and split-protocol changes before attributing a score difference to modeling.

Future evaluation should use several fully observed time windows, examine probability calibration, and compare contact-budget performance across subscription tiers. Gender is retained for historical feature parity; assess its necessity and subgroup behavior before any operational use. No causal retention uplift or production deployment is claimed.

## References

- [DuckDB memory management](https://duckdb.org/2024/07/09/memory-management)
- [scikit-learn decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn precision and recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
