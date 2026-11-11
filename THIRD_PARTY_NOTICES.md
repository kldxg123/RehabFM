# Third-party attribution

`third_party/sleepfm/models.py` and `base_config.json` originate from https://github.com/zou-group/sleepfm-clinical at commit `2bcbae04c3592f61352addb7ac3d4193f0a3ca25`. Copyright (c) 2025 Rahul Thapa. The upstream LICENSE is reproduced alongside the code: Creative Commons Attribution-NonCommercial 4.0, https://creativecommons.org/licenses/by-nc/4.0/. Upstream operator contents are preserved; any newline serialization is recorded in the source manifest. No endorsement is implied.

The project adaptation is in `research_archive/sleepfm_controlled_20261031_v13/study_models.py`. It resamples each original five-second channel to 640 samples, uses the released 128-wide/eight-head/six-layer operators, bridges 128 to 64 dimensions, maintains masks, substitutes a shape-safe layer-normalization invocation, and selects CUDA MATH SDPA for temporal pooling. These project modifications are distinct from the upstream source. The objective also draws on the 2024 SleepFM leave-one-out contrastive formulation (PMLR 235:48019–48037).

The archived DeepSense, LMF, MulT and UniTS adapters identify their original architectures in source and manuscript references. Their external implementation dependencies are not vendored by this release and must be obtained under their respective upstream licenses. The source archive is therefore not advertised as a self-contained reproduction environment for every comparator.

Public datasets are linked rather than redistributed. Neither this repository nor a clinical ethics approval changes third-party dataset access terms.
