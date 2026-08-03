# Round-one partial findings

The validation-only depth sweep was stopped on August 3, 2026. The official
MNIST test set was not evaluated. All results below use seed 7 and the same
stratified 12,000-train/2,000-validation split.

| Candidate | Status | Best epoch | Validation accuracy | Validation NLL | Mean depth | Forced exits |
|---|---:|---:|---:|---:|---:|---:|
| Depth 5, penalty `1e-3` | Complete | 17 | 90.60% | 0.308 | 5.0 | 100% |
| Depth 8, penalty `1e-3` | Complete | 18 | 89.80% | 0.347 | 8.0 | 100% |
| Depth 12, penalty `1e-3` | Paused after epoch 5 | 4 so far | 77.45% so far | 0.700 at epoch 4 | 12.0 | Not finalized |

The completed results do not support a benefit from increasing the horizon from
five to eight steps. Depth 8 trained more erratically, cost approximately 47%
more wall time, and finished 0.8 percentage points lower on validation. Both
completed routers sent every validation example to the horizon, so the current
compute penalty did not teach useful termination.

Routing is class-dependent, but usefulness is not established. For both depths,
the first route was usually `attention_016_h2`; digit 1 was usually sent first to
`attention_048_h6`. First-route/digit mutual information was 0.189 nats at depth
5 and 0.164 nats at depth 8. Both exceeded a 200-permutation shuffled-label null
at the resolution of that test (`p = 1/201`). This comparison does not yet include
the frozen-random-router control, so it is evidence of input-dependent routing,
not learned useful specialization.

Attention experts received 59.2% of visits at depth 5 and 76.9% at depth 8.
MLPs received 33.9% and 9.4%; convolutions received 7.0% and 13.7%. Several rare
experts had large digit-specific lift values, but their low absolute visit counts
make those lifts unstable. Interpret the visitation and lift heatmaps together.

The validation figures and machine-readable sources are under:

- `runs/round1/depth05_penalty1e3/seed_7/validation/analysis/`
- `runs/round1/depth08_penalty1e3/seed_7/validation/analysis/`

Resume instructions and hyperparameter locations are documented in
`docs/running_experiments.md`.

