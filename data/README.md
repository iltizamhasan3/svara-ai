# Week 2 datasets

This directory contains provenance for the Week 2 AI training and validation
inputs. Dataset payloads are downloaded locally into `data/raw/` and are
ignored by Git. The deterministic, small IGAR sample under `data/samples/` is
the only data payload intended to be tracked.

Run from the repository root:

```bash
python3 scripts/download_week2_datasets.py --help
python3 scripts/download_week2_datasets.py
```

The default output root is this `data/` directory. Use `--output-root` to
place the SmSA download elsewhere and `--igar-sample-size` to change the
sample size.
The tracked sample was generated with 30 rows (10 per source label); its
checksum and exact command are recorded in the manifest.
The downloader uses only Python standard-library modules, writes downloads
atomically, sends a descriptive User-Agent, verifies the pinned SmSA Git blob
hashes and the IGAR API-provided SHA-256/size, and validates TSV/CSV schemas
and labels.

## Sources and intended use

* **SmSA (SmSA document sentiment, IndoNLU)** is distributed with the
  Creative Commons Attribution Share-Alike 4.0 license recorded by the
  NusaCrowd adapter. Train, validation, and test files are pinned to commit
  `ce728f6926a36174b9923dfe49d6a6839b6e9bb7`.
* **IGAR (Indonesian Government App Review Dataset)** is published by Mahmud
  Isnan and Bens Pardamean on Mendeley Data, DOI
  `10.17632/7zryc6k76z.3`, under CC BY 4.0. The script obtains public
  metadata and file metadata from Mendeley, then streams `Rating_labeled.csv`
  from its public file URL only until the requested deterministic,
  label-stratified sample is complete. It does not download the approximately
  100 MB full file or the dataset ZIP.

IGAR is external/domain validation only. Its government-app review domain is
not a substitute for the primary SmSA training data and its sample must not be
presented as representative of all SVARA AI feedback. If a full IGAR file is
needed for a realistic external test, place it only at the ignored path
`data/raw/igar/Rating_labeled.csv` and acquire it with:

```bash
python3 scripts/download_igar_full.py
```

The full file must never be used for training, validation selection,
calibration, thresholding, or augmentation. Training data additions belong in
separate, provenance- and license-checked sources.

See `data/manifests/week2_sources.json` for exact URLs, expected hashes and
sizes, schemas, licenses, and provenance caveats.

The full IGAR SHA-256 and byte size in the manifest come from Mendeley's public
file metadata. Because the downloader intentionally stops after the requested
sample quotas, it does not claim to have rehashed the full multi-megabyte
source file locally.
