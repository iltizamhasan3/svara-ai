# Week 2 SmSA EDA artifacts

Generated from the pinned files under `data/raw/smsa/` with:

```bash
python3 scripts/prepare_week2_data.py
```

The JSON artifact keeps both views visible:

* `official_splits` reports the source files after within-split normalization
  and duplicate removal.
* `leakage_safe_splits` is the experiment view. The 14 normalized texts that
  appeared in both train and validation are removed from train while the
  official validation and test rows remain unchanged.

Observed baseline facts:

* source sizes are 11,000 train, 1,260 validation, and 500 test rows;
* 67 same-label duplicates are removed inside train;
* 14 train/validation overlaps are removed from the prepared train view;
* the resulting 10,919/1,260/500 split views have no normalized text overlap;
* train/validation are majority-positive (about 58%), while the test split is
  more balanced (41.6% positive, 40.8% negative, 17.6% neutral).

These are descriptive EDA results, not model performance. The official split
boundaries are retained, and the test split is not used for preprocessing or
hyperparameter selection.
