# Source archive map

`full_fixed_v1`: original backbone, task-local and adapted model comparisons, fixed readouts/predictions, scoring and statistical audit source.

`sleepfm_controlled_20261031_v13`: controlled three-backbone-seed comparison and stability/zero-loss evidence checks. The historical `official_source/sleepfm/models/models.py` dependency is distributed as `third_party/sleepfm/models.py`; the archive must be reconstructed in a separately authorized research workspace before execution.

`framework_revision_20261031`: six-task frozen-feature/readout-budget experiment.

`statistical_baselines.py`, `audit_raw_cohort.py`, `audit_native_raw_cohort.py`: descriptor baseline and cohort-audit source.

The portable Python package is the model dependency originally vendored by the fixed benchmark. Original synthetic demonstration training entrypoints are not the manuscript's formal trainers. `train_core.py` and `train_study.py` are the formal trainer source. Managers, launch guards, SSH settings, operational journals and full execution receipts are deliberately excluded.
