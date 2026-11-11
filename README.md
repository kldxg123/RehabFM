# RehabFM

Research code for **RehabFM: A Multimodal Data Fusion Platform for Rehabilitation Classification**.

RehabFM encodes heterogeneous EEG, EMG, ECG, EDA, IMU, force and skeleton inputs with modality-specific stems, channel pooling and a shared temporal encoder. Task-specific readouts use neural descriptors (F), statistical descriptors (S), or their concatenation (H). The repository documents the manuscript experiment, including negative results and implementation limitations.

## Quick start

Python 3.10 or newer is required. Install the PyTorch build appropriate for your device, then:

```sh
python -m pip install -e .
python examples/smoke_test.py
python tools/verify_release.py
```

The smoke test runs generated inputs on CPU, performs one optimizer update, and checks exact restoration. It uses no patient data and does not reproduce the paper's reported accuracies. Additional archived analysis scripts use `requirements.txt`; these dependency ranges are installation guidance, not an exact historical environment lock.

## Repository contents

* `src/rehabfm`: the manuscript backbone, tokenization, collation, loss functions and original synthetic demonstration utilities.
* `research_archive`: training, readout, prediction, statistical and audit source from the manuscript experiments. These preserve the research logic for inspection. Private workspace prefixes are replaced with placeholders; this folder is not an end-to-end public benchmark runner.
* `third_party/sleepfm`: pinned upstream reference operators and their original noncommercial license.
* `data_sources/datasets.csv`: all 22 public dataset sources plus the institutional cohort. Published source-cohort sizes are separate from retained project record counts.
* `results/manuscript_tables.json`: aggregate manuscript tables only; no individual predictions, labels, identifiers or features.
* `figures`: five monochrome SVG manuscript figures.
* `docs`: protocol, reproducibility boundaries, data access and third-party attribution.

The real-data training and fixed-test pipeline requires independently authorized datasets, local cohort materialization, split/target contracts, and private artifacts. Those artifacts and pretrained weights are not distributed. Public source availability does not imply that all clinical data or the complete private benchmark can be downloaded.

## Main results and interpretation

On the original 1,731-record external benchmark, RehabFM H classified 1,004 records correctly (58.00%); S classified 1,012 (58.46%). Across three backbone seeds, H averaged 58.58 ± 0.61% (sample SD). The adapted reference H averaged 57.85 ± 0.52%; their adjusted comparison was not significant. Neural-only F averaged 45.00 ± 2.19%, compared with 38.51 ± 0.44% for the adapted reference. The separate six-task 467-record frozen-feature experiment and the richer full-descriptor-bank results use different protocols and must not be pooled with the 1,731-record comparison.

The fixed benchmark had been inspected during historical development. These results are not evidence from an untouched prospective test set. Ten institutional test patients are not thirty independent patients when evaluated with three seeds. Component ablations do not establish a benefit for every architectural choice. CGMV-EGR has no measured comparison in this release. See `docs/PROTOCOL.md` for further qualifications.

## Attribution and license status

The leave-one-out objective and adapted reference operators draw on SleepFM. The reference experiment trains from scratch under common project objectives; it does not reproduce the original pretrained PSG system. See `THIRD_PARTY_NOTICES.md`. Upstream components retain their original licenses, including CC BY-NC 4.0 for the pinned reference source.

This is a public source release. No additional license grant for project-authored code or figures is made here pending the authors' institutional licensing decision. Do not describe the entire repository as MIT-licensed or as unrestricted commercial software. Dataset licenses remain with the original providers and do not transfer with this repository.

## Citation

The manuscript is under preparation; no journal acceptance or DOI is claimed. `CITATION.cff` records the research-code title and authors.
