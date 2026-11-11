# Manuscript experiment protocol and reproducibility boundaries

The release is scoped to the manuscript's RehabFM experiment, not subsequent TabICL/C2 or optimization work.

The original external benchmark contains 22 tasks and 1,731 query records. The institutional readout uses 19 training and ten test patients, with 281 and 163 records respectively. Only 17 tasks supply backbone gradients; 23 tasks have readouts. Source cohorts and retained records must not be confused with each other.

The controlled experiment uses five arms (full, no_condition, official_pool, ce_only, sleepfm_arch) and backbone seeds 0, 1101, 2202. There are 15 configurations: two original models are inherited and 13 new controlled models receive 60 epochs and 10,500 optimizer calls each. The comparison uses 690 F/H task results (92 original inherited and 598 newly produced). All predictions were sealed and restored before scoring. External and institutional paired analyses have separate families of ten Holm-corrected contrasts and 10,000 resamples/sign permutations. See the actual archived source for details.

The original component study uses a different 108-test external multiplicity family. These adjusted P values must not be mixed with the controlled study's ten-test families. The six-task supervised frozen-feature experiment uses 467 query records and readout sampling seeds; it is not zero-shot evaluation.

The reference is an operator adaptation trained under the common project objectives, not reproduction of a pretrained PSG system or the original LOO-only training. Parameters and compute are not matched. Reference seed 0 retains epochs 1–2 using the old temporal attention backend and switches to CUDA MATH SDPA from epoch 3; the other two reference seeds use the repaired backend from initialization. Rounding and dropout random mappings can differ across backends. No cross-backend bitwise-equivalence claim is made.

The no-condition variant retains the token slot and channel coding. The reference-pooling variant retains the 64-wide/four-head project context. CE-only uses coefficient 1 while full uses 0.3, so it is not a single-factor auxiliary-term-count comparison. Structural zero gradients are accepted only with independent proof that every weighted loss is unavailable or identically zero; unexplained zero gradients, nonfinite values and incomplete budgets remain failures.

Historical fixed-test exposure, unresolved source identities/archive-version mappings, small institutional sample size, and absence of an untouched prospective cohort limit interpretation. No measured CGMV-EGR result is available. The overall hybrid model does not significantly outperform the statistical-only control in the reported controlled comparison.

## What this public package reproduces

The installed package and generated-input test reproduce the released backbone's execution and state restoration. Aggregate-table arithmetic can be checked without datasets. Scientific training, fixed predictions and clinical results require private inputs and the recorded research environment. Historical scripts retain freeze checks, private-artifact expectations and device gates; do not remove these assertions and call a resulting run an exact reproduction. Path sanitization and encoding changes are logged in `SOURCE_MANIFEST.json`.

No new clinical training, predictions or outcome validation were performed to prepare this release. The generated-input test is software verification only.
