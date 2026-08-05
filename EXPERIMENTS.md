# MNIST SGLR experiment ledger

Updated: 2026-08-05. Workstation: `ubuntu@192.222.50.154`, NVIDIA GH200 96 GB,
Python 3.10.12, PyTorch 2.7.0, CUDA available. MNIST is cached before timed runs.

## Fixed acceptance gates

- Data: 60,000 official training examples and a deterministic, stratified 5,000/5,000
  split of the official test set for validation and sealed testing.
- Selection: save the earliest strict validation-accuracy maximum and stop after the
  configured patience. Evaluate the sealed test once with those saved weights.
- Accuracy: more than 98% on exactly 5,000 sealed test examples.
- Experts: exactly 24, comprising 8 MLP, 8 attention, and 8 convolution experts.
- Evaluation: top-1 sequential sparse routing, with one selected expert per active
  example and recurrent step.
- Time: a fresh, data-cached CLI process must finish in at most 180 seconds. Record
  training and end-to-end wall time separately.
- Differential use: digit/first-route mutual information at least 0.05 nats with a
  1,000-permutation p-value at most 0.05.
- Even use: all 24 experts visited, utilization entropy divided by `log(24)` at least
  0.95, and every visit share between `0.5 / 24` and `2 / 24`.
- Termination: more than 90% natural exits, equivalent to `forced_exit_rate < 0.10`.
- Depth: mean routed depth from 5 through 10 experts, added after E4F at the user's
  request.

Validation-only trials do not access the sealed 5,000-example test partition.

## Experiments

### P0: environment and pipeline smoke

- Command: `python3 -m scripts.smoke_test`, then `python3 -m scripts.train_mnist
  --preset smoke --device cuda --download --no-progress`.
- Result: synthetic smoke passed in 1.24 seconds. The end-to-end CUDA smoke passed in
  5.12 seconds including the one-time MNIST download.
- Decision: cache MNIST and start full-data timing. The smoke metrics are not evidence
  for an acceptance gate because it used 256 training examples.

### E0: current `diverse_full` baseline

- Configuration: straight-through routing; 24 experts; depth 1 to 20; 15 epochs;
  batch 256; learning rate `1e-3`; balance `0.03`; route MI `0.1`; compute penalty
  `0.025`; 60,000/5,000/5,000 split.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset diverse_full
  --device cuda --no-progress`.
- Result: failed the runtime gate. The process was terminated at 180.4 seconds before
  epoch 1 completed and wrote only `manifest.json`.
- Interpretation: dense straight-through training computes every expert at every
  active step. Depth 20 is not viable under the per-run limit.
- Next: use sparse training with a shallow route and validation-only candidate runner.

### E1: current full model with sparse hard-argmax training

- Configuration: hard-argmax routing; 24 experts; depth 1 to 5; 20 epochs; batch
  256; learning rate `3e-4`; balance `0.01`; compute penalty `0.001`;
  60,000/5,000/5,000 split. The run evaluated validation only.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset full --variant
  hard_argmax --device cuda --validation-only --no-progress`.
- Result: failed the runtime gate at 180.4 seconds after three complete epochs.
  Validation accuracy was 22.34%, 26.46%, and 31.60%. Validation mean depth fell
  from 4.70 to 1.85 to 1.44.
- Interpretation: sparse dispatch alone is insufficient with batch 256 and depth 5.
  The compute objective can shorten routes, but the shared image model learns slowly.
- Next: use depth 1 to 2, batch 2,048, a larger learning rate, and stronger routing
  losses before changing the encoder.

### E2: shallow large-batch sparse routing

- Configuration: hard-argmax routing; 24 experts; depth 1 to 2; 12 epochs; batch
  2,048; learning rate `2e-3`; balance `0.1`; route MI `0.05`; compute penalty
  `0.5`; validation-only evaluation.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset fast_full
  --device cuda --validation-only --no-progress`.
- Result: completed in 79.81 seconds externally and 78.56 seconds internally. The
  validation-selected epoch was 10 of 12 at 42.74% accuracy. Forced-exit rate was
  0%. All experts were used; shares ranged from 2.96% to 6.04%, and normalized
  utilization entropy was 0.996. Digit/first-route MI was 0.456 nats with
  permutation `p=0.001`.
- Interpretation: time, termination, balance, and digit differentiation passed. The
  weak image encoder and mean-pooled linear readout limit accuracy.
- Next: add a small shared convolutional stem and spatial MLP readout. Keep routing,
  losses, depth, and data protocol fixed to isolate the accuracy change.

### E3: shared CNN stem and spatial MLP readout

- Configuration: E2 routing and training settings; shared convolutional encoder width
  32; spatial readout hidden size 128; 451,613 total parameters; validation-only
  evaluation.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset fast_cnn_full
  --device cuda --validation-only --no-progress`.
- Result: completed in 65.08 seconds externally and 63.85 seconds internally. Patient
  stopping selected epoch 7 at 98.44% validation accuracy and stopped before epoch 11.
  Forced-exit rate was 0%. All experts were used; shares ranged from 2.10% to 5.88%,
  and normalized utilization entropy was 0.990. Digit/first-route MI was 0.637 nats
  with permutation `p=0.001`.
- Interpretation: every validation gate passed. Freeze this configuration and perform
  one fresh deterministic run with a single sealed-test evaluation.

### E3F: frozen E3 run with sealed test

- Configuration: identical to E3, retrained from scratch. The saved checkpoint was
  selected at epoch 7 by validation accuracy and patience 3.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset fast_cnn_full
  --device cuda --no-progress`.
- Result: completed in 64.92 seconds externally and 63.70 seconds internally. Test
  accuracy was 98.22% on 5,000 examples. Forced-exit rate was 0%. All 24 experts
  were used, normalized utilization entropy was 0.988, and digit/first-route MI was
  0.612 nats with permutation `p=0.001`. Shares ranged from 1.76% to 6.24%.
- Interpretation: accuracy, time, termination, entropy, and digit differentiation
  passed. The 1.76% minimum share missed the predeclared 2.08% lower bound.
- Next: increase both total and within-family balance pressure, then require a safer
  validation utilization margin before another final evaluation.

### E4: stronger hierarchical balance

- Configuration: E3 plus load-balance coefficient `0.2` and within-family balance
  weight `2.0`; validation-only evaluation.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_balanced_full --device cuda --validation-only --no-progress`.
- Result: completed in 78.39 seconds externally and 77.15 seconds internally. The
  validation-selected checkpoint was epoch 12 at 98.52% accuracy. Forced-exit rate
  was 0%. All experts were used; shares ranged from 3.32% to 4.92%, and normalized
  utilization entropy was 0.998. Digit/first-route MI was 0.633 nats with permutation
  `p=0.001`.
- Interpretation: every validation gate passed with a wide utilization margin. Freeze
  this configuration for the final end-to-end test run.

### E4F: accepted final run

- Configuration: identical to E4, retrained from scratch; evaluation used the epoch-12
  checkpoint selected by validation accuracy with patience 3.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_balanced_full --device cuda --no-progress`.
- Result: completed in 77.98 seconds externally and 76.74 seconds internally. Test
  accuracy was 98.62% on exactly 5,000 examples. Forced-exit rate was 0%. All 24
  experts were used; shares ranged from 3.34% to 5.06%, and normalized utilization
  entropy was 0.998. Digit/first-route MI was 0.606 nats with permutation `p=0.001`.
- Verification: `python3 -m scripts.verify_acceptance --run-dir
  runs/fast_cnn_balanced_full/hard_argmax/seed_7` reported all nine gates as passed.
  The final remote test suite passed 62 tests in 4.32 seconds; the sparse-dispatch
  test instruments expert forward calls and confirms that evaluation invokes only the
  selected expert for each active example.
  The saved `best_model.pt` SHA-256 is
  `758288dad56daaa6e2375e30e24ffa5d73b4b2747058f1daf657ff5451f14c77`.
- Artifact: `runs/fast_cnn_balanced_full/hard_argmax/seed_7`.

E4F passed the original gates but is superseded because its mean routed depth was 1.0.

### E5: five full-strength expert updates

- Configuration: minimum depth 5, maximum depth 6, batch 4,096, 8 epochs, learning
  rate `3e-3`, patience 2, and E4 losses.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_full --device cuda --validation-only --no-progress`.
- Result: completed in 152.36 seconds externally and 151.10 seconds internally.
  Validation accuracy was 94.18% at epoch 8 and mean depth was 5.27. Forced-exit
  rate was 26.72%. Utilization ranged from 1.55% to 7.78%, with normalized entropy
  0.970.
- Interpretation: depth and time passed. Accuracy, termination, and minimum
  utilization failed. Repeated full-strength expert residuals degraded the shared
  representation, and the exit gradient was diluted across six route decisions.
- Next: scale expert residuals to 0.1, increase balance to `0.5`, and increase the
  compute penalty to `3.0` so the final exit decision receives similar gradient
  pressure to the depth-1 model.

### E6: scaled residuals and stronger routing losses

- Configuration: E5 with expert residual scale `0.1`, load balance `0.5`, and compute
  penalty `3.0`.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_scaled_full --device cuda --validation-only --no-progress`.
- Result: completed in 158.13 seconds externally and 156.85 seconds internally.
  Validation accuracy was 97.08% at epoch 8, mean depth was 5.008, and 99.18% of
  examples exited naturally. Utilization ranged from 2.28% to 5.35%, with normalized
  entropy 0.994.
- Interpretation: time, depth, termination, and utilization passed. Residual scaling
  recovered 2.9 accuracy points but remained 0.92 points below the validation target.
- Next: use four shallow routing warmup epochs followed by six full depth-5 epochs in
  one timed run. Validation remains full-depth throughout.

### E7: shallow-to-depth-5 routing curriculum

- Configuration: E6 with 10 total epochs, four one-expert training warmup epochs,
  six depth-5 training epochs, and patience 3. Validation always used full depth 5.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_curriculum_full --device cuda --validation-only --no-progress`.
- Result: completed in 144.31 seconds externally and 143.05 seconds internally.
  Validation accuracy was 98.18% at epoch 8, mean depth was 5.076, and 92.42% of
  examples exited naturally. Utilization ranged from 0.21% to 11.04%, with normalized
  entropy 0.922.
- Interpretation: accuracy, time, depth, and termination passed. Utilization failed
  because source-specific downstream routers received training only after warmup.
- Next: share one input-dependent router across all recurrent sources so every epoch
  trains the router used at every depth.

### E8: shared router across recurrent sources

- Configuration: E7 with one input-dependent router reused at all recurrent depths.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_shared_router_full --device cuda --validation-only --no-progress`.
- Result: completed in 96.52 seconds externally and 95.29 seconds internally.
  Validation accuracy was 98.28% at epoch 9, mean depth was exactly 5.0, and every
  example exited naturally. Digit/first-route MI was 0.455 nats with permutation
  `p=0.001`. Utilization ranged from 0.52% to 10.53%, with normalized entropy 0.952.
- Interpretation: accuracy, time, depth, termination, and digit differentiation
  passed. Hard utilization failed even though mean router-probability entropy was near
  its maximum; small logit differences were amplified by argmax.
- Next: use differentiable Sinkhorn balancing over each batch before the five expert
  argmax decisions. Keep the learned exit decision unchanged.

### E9: differentiable Sinkhorn balancing

- Configuration: E8 with 100 Sinkhorn iterations at temperature `0.05` before each
  of the five expert decisions.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_sinkhorn_full --device cuda --validation-only --no-progress`.
- Result: completed in 162.43 seconds externally and 161.16 seconds internally.
  Validation accuracy was 98.24% at epoch 8, mean depth was exactly 5.0, and every
  example exited naturally. Digit/first-route MI was 0.699 nats with permutation
  `p=0.001`. Hard utilization ranged from 1.00% to 10.71%, with normalized entropy
  0.941.
- Interpretation: Sinkhorn balanced differentiable probability mass but did not
  guarantee balanced discrete argmax assignments.
- Next: retain Sinkhorn training and use score-preserving capacity-constrained top-1
  assignment during validation and test. Each expert receives floor or ceiling batch
  capacity, while sparse one-expert execution remains unchanged.

### E10: capacity-constrained top-1 evaluation

- Configuration: E9 plus score-preserving capacity-constrained top-1 assignment during
  validation and test. Training remains differentiable Sinkhorn hard routing.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_capacity_full --device cuda --validation-only --no-progress`.
- Result: completed in 157.18 seconds externally and 155.92 seconds internally.
  Validation accuracy was 98.34% at epoch 10, mean depth was exactly 5.0, and every
  example exited naturally. Expert shares ranged from 4.14% to 4.18%, with normalized
  entropy effectively 1.0. Digit/first-route MI was 0.751 nats with permutation
  `p=0.001`.
- Interpretation: every validation gate passed, including the added depth target.
  Freeze this configuration for a fresh final held-out run.

### E10F: accepted depth-5 final run

- Configuration: identical to E10, retrained from scratch. Validation accuracy and
  patience selected the epoch-10 checkpoint before held-out evaluation.
- Command: `timeout 180s python3 -m scripts.train_mnist --preset
  fast_cnn_depth5_capacity_full --device cuda --no-progress`.
- Result: completed in 158.33 seconds externally and 157.05 seconds internally. Test
  accuracy was 98.24% on exactly 5,000 examples. Mean depth was exactly 5.0 and every
  example exited naturally. All 24 experts were used; shares ranged from 4.14% to
  4.18%, and normalized utilization entropy was 0.999997. Digit/first-route MI was
  0.726 nats with permutation `p=0.001`.
- Verification: `python3 -m scripts.verify_acceptance --run-dir
  runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7` reported all ten gates as
  passed. The remote suite passed 79 tests. The saved `best_model.pt` SHA-256 is
  `f639e70190a8ead29a7ee625bda5a9eda1c42649b508f01fd5136b3445b51016`.
- Artifact: `runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7`.
