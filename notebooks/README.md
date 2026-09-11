# AI experiment notebooks

`week2_indobert_experiment.ipynb` is the Week 2 preparation notebook for the
SmSA sentiment baseline. It loads the pinned official train/validation/test
files, applies `preprocessing-v1`, removes training text that overlaps with
validation or test, and contains an opt-in IndoBERT fine-tuning cell.

The notebook intentionally sets `RUN_TRAINING = False`. Set it to `True` only
after installing the optional AI dependencies and when a model download and
training run are desired:

```bash
cd backend
python -m pip install -e '.[data,ai]'
jupyter lab ../notebooks/week2_indobert_experiment.ipynb
```

Training metrics and the confusion matrix are written to `artifacts/week2/`
by the notebook. The exact checkpoint and hyperparameters remain candidates
for Week 3 selection; the test split is never used for tuning. IGAR is loaded
only as a clearly labelled external/domain-validation sample.
