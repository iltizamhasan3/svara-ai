# SVARA AI — Sentiment Model v2.0.0

Status: released as an opt-in model bundle. The v1 release and its trust manifest remain available and unchanged.

## Release

- GitHub release: [`ai-model-v2.0.0`](https://github.com/iltizamhasan3/svara-ai/releases/tag/ai-model-v2.0.0)
- Asset: `svara-ai-sentiment-model-v2.tar.gz`
- Runtime manifest: [`sentiment_model_export_manifest_v2.json`](../artifacts/week5/sentiment_model_export_manifest_v2.json)
- Model contract: `sentiment-model-v2`
- Base model revision: `indobenchmark/indobert-base-p1`, revision `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2`
- Labels: `positive=0`, `neutral=1`, `negative=2`

## Training provenance

The model starts from the v1 SmSA model and trains only the classifier head on 4,208 disjoint Indonesian Google Play Review rows (the published train and validation partitions combined). The Google Play source is used as a domain-adaptation source and is licensed CC BY 4.0.

- epochs: `1.0`
- learning rate: `1e-4`
- batch size: `64`
- max sequence length: `128`
- seed: `42`
- device: CPU, PyTorch threads `4`
- encoder: frozen
- IGAR labels and metrics: not used for training, tuning, or checkpoint selection

## Validation evidence

| Holdout | v1 accuracy | v1 macro-F1 | v2 accuracy | v2 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| SmSA validation | 0.937302 | 0.910796 | 0.938095 | 0.911270 |
| Google Play confirmation | 0.769723 | 0.575428 | 0.826241 | 0.593704 |

The Google Play labels are derived from published star ratings, so this improvement is evidence for the app-review proxy domain, not a general guarantee.

## External-test caveat

The frozen IGAR test result for this candidate was `0.6667` accuracy, while the previously retained v1 ensemble reached `0.7000`. Therefore v2 is released for reproducible opt-in evaluation and domain experimentation; it does not replace v1 as the default or claim overall superiority on IGAR.

## Use

Extract the release asset and pass the v2 model directory together with the v2 export manifest to the local inference command. The loader remains local-files-only and validates all required file checksums.
