# SGLR: Self-Gated Latent Reasoners

SGLR is a small research system for testing whether a neural network can learn how to route each input through a collection of heterogeneous modules. The first experiment is deliberately narrow: MNIST images, 24 tiny experts, at most five routed expert updates, and route traces that can be compared across digits and controls.

The central question is not whether SGLR can classify MNIST. Ordinary networks already do that well. The question is whether task-trained, discrete routing discovers repeatable and useful class-dependent computation while remaining competitive at a comparable parameter and compute budget.

## Accepted full-data result

The accepted seed-7 run of `fast_cnn_depth5_capacity_full` reached **98.24% test accuracy** on 5,000 held-out MNIST images. The epoch-10 checkpoint was selected by a validation accuracy of 98.34% with patience 3, then evaluated once on the held-out test half. One complete training and evaluation run took 157.04958913799783 seconds internally and 158.33 seconds by `/usr/bin/time` on the target NVIDIA GH200 96 GB workstation, below the 180-second limit. MNIST was cached before the timed run.

The data protocol uses all 60,000 official training images. A deterministic, stratified split of the official 10,000-image test set supplies 5,000 validation images for checkpoint selection and 5,000 disjoint images for final testing.

| Result | Exact value |
| --- | ---: |
| Best validation accuracy and epoch | 98.34%, epoch 10 |
| Held-out test accuracy | 98.24% (4,912/5,000) |
| Held-out negative log-likelihood | 0.057732065582275394 |
| Mean and 95th-percentile route depth | 5.0 and 5 experts |
| Natural termination | 100%; forced-exit rate 0.0% |
| Expert visits | 1,035 to 1,045 of 25,000; 4.14% to 4.18% per expert |
| Normalized utilization entropy | 0.9999967744488256 |
| Digit/first-route mutual information | 0.7264983343405851 nats |
| Permutation test | `p=0.000999000999000999` with 1,000 label permutations |
| Trainable parameters | 420,293 |
| Mean routed-expert compute | 954,158.184 analytical MACs per image |
| Training throughput and peak CUDA memory | 3,869.692831462357 images/s; 3,928,611,840 bytes |
| Test throughput | 7,295.052137719451 images/s |

The exact winning configuration is:

| Component | Setting |
| --- | --- |
| Representation | width-32 CNN encoder; 4 x 4 patches on a 7 x 7 grid; 49 tokens of width 48; 128-unit spatial readout |
| Router | shared across recurrent sources; hidden size 16; dropout 0; `hard_argmax` routing |
| Route depth | `min_steps=5`, `max_steps=6`; expert residual scale 0.1 |
| Expert pool | 24 experts: 8 MLP, 8 attention, and 8 convolution |
| MLP expert widths | 8, 12, 16, 24, 32, 48, 64, 96 |
| Attention expert `(internal width, heads)` | (8, 1), (12, 1), (16, 1), (16, 2), (24, 2), (24, 3), (32, 4), (48, 6) |
| Convolution expert `(kernel, dilation, channels)` | (3x1, 1x1, 2), (5x1, 1x1, 2), (1x3, 1x1, 2), (1x5, 1x1, 2), (3x3, 1x1, 2), (5x3, 1x1, 3), (3x1, 2x1, 2), (3x3, 2x1, 3) |
| Batch balancing | 100 Sinkhorn iterations at temperature 0.05 during the five mandatory routing decisions; capacity-constrained top-1 assignment during validation and test |
| Training schedule | 10 epochs; batch size 4,096; first 4 epochs at depth 1, final 6 at depth 5; full-depth validation throughout |
| Optimizer | AdamW; learning rate 0.003; weight decay 0.0001; linear warmup fraction 0.02; gradient accumulation 1 |
| Auxiliary loss coefficients | load balance 0.5; within-family balance 2.0; route mutual information 0.05; compute penalty 3.0 |
| Selection and reproducibility | patience 3; seed 7; CUDA; no parameter-budget constraint |

Validation and test proceed through recurrent steps sequentially and execute only one assigned expert per active example at each step. Even utilization is therefore an enforced evaluation constraint, not a purely emergent property of unconstrained argmax routing. Within that constraint, a greedy score-ranked assignment uses learned router scores, and the reported mutual information and permutation test quantify digit differentiation.

Run the accepted configuration and verify every acceptance gate with:

```bash
timeout 180s python3 -m scripts.train_mnist --preset fast_cnn_depth5_capacity_full --device cuda --no-progress
python3 -m scripts.verify_acceptance --run-dir runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7
```

The [acceptance report](runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7/acceptance.json), [resolved manifest](runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7/manifest.json), [evaluation summary](runs/fast_cnn_depth5_capacity_full/hard_argmax/seed_7/evaluation_summary.json), and [complete experiment ledger](EXPERIMENTS.md) record the result. All ten acceptance gates passed, and the remote test suite passed 79 tests. The selected `best_model.pt` has SHA-256 `f639e70190a8ead29a7ee625bda5a9eda1c42649b508f01fd5136b3445b51016`.

## Model

The MNIST model converts an image into a Transformer-shaped latent and passes that latent through a reusable recurrent core.

```text
image (b, 1, 28, 28)
  -> learned 4 x 4 patch embedding
tokens X_0 (b, 49, 48)
  -> SGLRCore, one to five routed expert updates
tokens X_T (b, 49, 48)
  -> LayerNorm -> masked mean pool -> linear classifier
logits (b, 10)
```

At recurrent step `t`, the router associated with the current source pools the tokens and predicts one of 24 experts or the exit action. There is one initial router and one router after each expert, so self-loops and transitions to every other expert are available:

$$
p_t = \mathrm{softmax}(R_{\mathrm{source}(t)}(\mathrm{pool}(X_t))).
$$

If expert `e` is selected, it produces a delta with the same shape as its input. The recurrent core owns the only residual update:

$$
\Delta_t = E_e(\mathrm{LayerNorm}(X_t)), \qquad
X_{t+1} = X_t + \Delta_t.
$$

Exit passes the latent unchanged to the classifier. Exit is unavailable until one expert has run. Samples that have not selected exit after five expert steps are marked as forced exits rather than being silently conflated with learned termination.

The expert families impose different, simple priors:

- **MLP experts** independently mix the 48 features within each token.
- **Attention experts** mix information across the 49-token sequence and honor the attention mask.
- **Convolution experts** view `(l, d)` as a two-dimensional activation map and use small odd, same-padded kernels. Token-only, feature-only, joint, and dilated kernels provide distinct local biases.

The original primary preset contains eight experts from each family. Their widths, kernels,
channels, heads, and dilation values are generated in `sglr/presets/mnist.py`.
That preset must remain below 200,000 total parameters, and its complete
router pool must not contain more parameters than its complete expert pool. Both
invariants are checked for the canonical 24-expert pool. The accepted full-data
preset reported above intentionally relaxes both constraints and has 420,293
parameters.

## Discrete routing and credit assignment

The original primary model uses a straight-through top-1 gate during training. Its forward pass is discrete, while its backward pass uses the softmax gradient:

```text
probabilities = softmax(router(pooled_tokens))
hard_gate = one_hot(argmax(probabilities))
gate = hard_gate + probabilities - stop_gradient(probabilities)
```

All expert candidates are computed for active samples during straight-through training. The unchanged latent is included as the exit candidate, allowing classification loss to teach both routing and termination. Evaluation executes only the selected expert, producing sparse and unambiguous traces.

The accepted configuration instead uses sparse hard-argmax training. Sinkhorn-balanced probabilities supply gradients only through the auxiliary routing objectives; classification gradients do not pass through the discrete route choices.

Two auxiliary objectives are intentionally limited in scope. Load balancing is computed over expert actions only, never exit. A small differentiable compute penalty discourages unnecessary depth. Neither objective is evidence that a route is useful; usefulness must appear in held-out task performance and comparisons with the controls below.

## Experiments

The focused sweep runs three seeds (`7`, `17`, and `27`) for four variants:

1. `straight_through`: independent task-trained routers with straight-through top-1 gates.
2. `hard_argmax`: the same pool and hard choices, but no task gradient through route decisions.
3. `frozen_random`: input-dependent routers initialized randomly and then frozen.
4. `fixed_depth`: five residual blocks in the sequence MLP, convolution, attention, convolution, MLP.

The fixed-depth baseline adjusts its final MLP width to the smallest multiple of eight that places its trainable parameter count within 2% of the gated model. This control is important: accuracy differences are not interpretable if one model simply receives a substantially larger parameter budget.

The supplied Python presets are:

- `smoke`: 256 training examples, 128 validation examples, and one epoch.
- `pilot`: a seeded stratified 12,000/2,000 train/validation subset and five epochs.
- `full`: a fixed 50,000/10,000 split of the official MNIST training set.
- `focused`: the four-variant, three-seed pilot sweep used for the primary comparison.
- `fast_cnn_depth5_capacity_full`: the accepted full-data, depth-5 configuration reported above.

These are ordinary Python functions in `sglr/presets/mnist.py`. To expand the
pool uniformly, pass `--experts-per-family 32`, or call `make_expert_pool()` with
separate MLP, attention, and convolution counts from Python.

For the accepted preset, the validation half of the official test split is used for checkpoint selection, while its disjoint held-out half is evaluated once after training completes. Subset selection is seeded and stratified.

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -r requirements.txt
```

The environment contains PyTorch, torchvision, NumPy, Matplotlib, tqdm, pytest,
Ruff, and mypy. Historical Streamlit, graph, and probe workflows are not dependencies.

## Commands

Train one variant:

```bash
python -m scripts.train_mnist --preset pilot --variant straight_through --seed 7
```

Run or resume the focused sweep:

```bash
python -m scripts.run_mnist_sweep --preset focused
```

Rebuild one run's figures from frozen evaluation artifacts without training:

```bash
python -m scripts.analyze_mnist run --run-dir runs/<completed_run>
```

Recompute the evaluation artifacts and figures from a frozen checkpoint:

```bash
python -m scripts.analyze_mnist checkpoint --run-dir runs/<completed_run>
```

Aggregate the completed focused sweep:

```bash
python -m scripts.analyze_mnist sweep --sweep-root runs/focused --output-dir figures/generated/focused
```

CLI flags use kebab-case; typed Python preset functions are the single source for experiment defaults.
Training commands show nested run, epoch, batch, validation, and evaluation
progress with live loss, accuracy, route depth, learning rate, throughput, and
early-stopping state. Add `--no-progress` for CI or plain redirected logs.
MNIST subsets are normalized and materialized as contiguous tensors once when a
run starts. Training uses pinned host memory only for CUDA runs, and configured
workers remain alive across epochs. Evaluation transfers one compact batch trace
to the CPU rather than synchronizing the GPU for every example and route.
Use `--experts-per-family 16` for 48 experts or edit
`sglr/presets/mnist.py` to give the MLP, attention, and convolution families
different counts. The canonical 24-expert order remains stable so old
checkpoints and recorded route IDs retain their meaning. Expanded presets get a
distinct run name automatically, such as `pilot_48_experts`.

For validation-only depth tuning, safe resume behavior, and a field-by-field
hyperparameter guide, see [`docs/running_experiments.md`](docs/running_experiments.md).

## Run artifacts

Each run directory is self-contained and schema-versioned. Its analysis contract is:

- `manifest.json`: the resolved configuration, schema version, command, seed, Git commit and dirty state, device, package versions, elapsed time, throughput, peak CUDA memory, and checkpoint locations.
- `evaluation_summary.json`: parameter counts, analytical expert compute, accuracy, NLL, confusion matrix, route statistics, and forced-exit rate.
- `evaluation.jsonl`: one test record per image with label, prediction, confidence, correctness, ordered expert IDs, depth, and forced-exit status.
- `evaluation_images.npz`: optional images and sample indices used for representative route panels.
- `best_model.pt` and `last_state.pt`: the best validation checkpoint and resumable optimizer, scheduler, RNG, and loader state.
- `run_complete.json`: written last, after the manifest and evaluation artifacts are complete.
- `analysis/`: regenerated 300 dpi figures and their CSV or JSON source values.

Summary and figure generation consume these recorded evaluation artifacts. They do not silently retrain models.
The compute proxy is the analytical number of multiply-accumulates in the experts actually executed by each sample. It includes expert projections, attention matrix products, and convolutions, but excludes the shared encoder, routers, normalization, classifier, nonlinearities, and hardware overhead. It is recorded separately from wall time and is not an exact latency measurement.

## Figure guide

The figure command writes 300 dpi PNG files alongside the CSV or JSON values used to construct them:

- **Accuracy versus mean expert compute** compares variants across seeds. Error bars show between-seed standard deviation.
- **Digit-by-expert visitation** shows how often each digit uses each expert. The specialization-lift companion divides this rate by the expert's marginal visitation rate.
- **Digit-by-first-route** isolates the first routing choice, before routes branch recursively.
- **Digit-by-exit-depth** and **route-depth distributions** show learned termination and forced exits by class.
- **Expert transition heatmap** counts consecutive routed expert pairs.
- **Representative route panels** pair a median-confidence image per digit with its ordered expert trace, preferring correct predictions when available; `representative_routes.csv` records the selections.
- **Route/digit mutual information** is reported against shuffled-label permutations. A positive raw value without separation from this null is not evidence of specialization.

Training curves are diagnostic, not a primary result. Exact route strings are not used as the main aggregate because long routes fragment into sparse, unreadable categories.

## Interpreting results

Digit-conditioned routes are expected even from an untrained input-dependent router because different digits have different pixel distributions. Accordingly, a visually attractive heatmap is not enough. A specialization claim requires class-conditioned routing statistics to exceed shuffled-label and frozen-random controls across seeds, while the straight-through model also improves or preserves held-out performance at comparable parameter and compute cost.

The hard-argmax ablation tests the specific value of classification gradients reaching route decisions. The fixed-depth model tests whether recurrence and conditional computation add value beyond an ordinary sequence of heterogeneous residual blocks. Negative results are still informative: they may show that the routing mechanism adds complexity without useful organization at this scale.

## Historical material

Earlier code explored large flat-vector experts, probe heads, a Streamlit route viewer, graph residence-time simulations, and speculative future directions. Those workflows and their dependencies are historical context, not part of the focused experiment. Their routers were not trained through task loss, so prior digit-specific route plots cannot be treated as evidence that SGLR learned useful specialization. See `archive/README.md` for the archival map and limitations.
