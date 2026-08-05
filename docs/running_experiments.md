# Running and modifying SGLR experiments

The paused round-one results are summarized in
[`round1_partial_findings.md`](round1_partial_findings.md).

## Resume the first validation-only depth round

The first round is resumable. Completed candidates are skipped, and an interrupted
candidate resumes from its last complete epoch:

```bash
python -m scripts.run_round1 \
  --preset pilot \
  --output-root runs/round1 \
  --device cuda \
  --epochs 20 \
  --patience 5
```

Candidate selection uses only the 2,000-example validation subset. The script
evaluates the official test set once, after all candidates finish and the winner
is frozen. If `runs/round1/selected_test/run_complete.json` exists, rerunning the
command does not evaluate test again.

The script records `status: test_started` before loading the sealed test split.
If evaluation is interrupted after that point, it fails closed on the next run
instead of silently testing twice. Inspect the saved artifacts and use the
offline analysis command to regenerate any missing figures.

Use a new `--output-root` whenever you change the candidate grid or selection
protocol. This prevents an old sealed-test result from being associated with a
new validation sweep.

## Change the round-one hyperparameters

The candidate grid is the `ROUND_ONE_CANDIDATES` tuple near the top of
`scripts/run_round1.py`. Each entry has:

```python
RoundOneCandidate(
    name="depth12_penalty3e4",
    max_steps=12,
    min_steps=1,
    learning_rate=1e-3,
    load_balance_coefficient=0.01,
    compute_penalty_coefficient=3e-4,
)
```

- `max_steps` is the recurrent horizon. Training cost grows roughly linearly
  with active depth because straight-through training evaluates every expert
  candidate at each step.
- `min_steps` is the number of expert applications required before exit becomes
  available.
- `learning_rate` is the AdamW peak learning rate before cosine decay.
- `load_balance_coefficient` controls the expert-only balancing loss.
- `compute_penalty_coefficient` penalizes probability mass assigned to continued
  computation. Larger values encourage earlier exits.

The command-line `--epochs` and `--patience` flags control the maximum schedule
and validation-accuracy early stopping. `--device` accepts `auto`, `cpu`, or
`cuda`. Runs display nested candidate, epoch, and batch progress bars. The live
postfix reports loss, accuracy, route depth, learning rate, best validation
accuracy, and remaining early-stopping patience. Evaluation bars report running
accuracy, NLL, route depth, and throughput. Use `--no-progress` to disable bars
without changing the resolved experiment or its resume path.

### Throughput notes

MNIST is small enough to keep in system memory. The loader therefore normalizes
each selected image once at startup and defaults to `num_workers=0`; extra worker
processes add more coordination than useful preprocessing on this workload. If
you explicitly set a positive worker count, the training and validation workers
remain alive between epochs, and CUDA runs use pinned-memory transfers.

On the local RTX 3070 used for development, materializing the 12,000-example
pilot subset reduced one data-only epoch from 2.39 seconds to 0.32 seconds. For
the canonical depth-five model, batch sizes 128, 256, 512, and 1,024 delivered
approximately 570, 900, 1,050, and 1,070 training examples per second. Batch
1,024 used about 5.8 GiB, so 512 is the practical depth-five choice on an 8 GiB
card, while 256 leaves safer headroom for the deeper round-one candidates.

Batch size changes the number of optimizer updates and is therefore an
experimental hyperparameter, not merely a systems setting. Give a changed batch
size a distinct preset name and confirm validation accuracy rather than mixing
it into an existing run directory.

## Change the model and data preset

The functions in `sglr/presets/mnist.py` are the source of the common model,
expert-pool, optimizer, and data settings. Available preset names are `smoke`,
`pilot`, `full`, `diverse_full`, and `focused`.

### Hierarchical balance and route diversity run

`diverse_full` is the first preset that jointly enables the hierarchical
hard-soft Switch loss and per-step routing mutual information:

```python
TrainingConfig(
    load_balance_coefficient=0.03,
    within_family_balance_weight=1.0,
    route_mi_coefficient=0.1,
    compute_penalty_coefficient=0.025,
)
```

The balancing objective first balances hard and soft utilization across MLP,
convolution, and attention families, then balances experts within every
represented family. Exit decisions are excluded. The two levels are normalized
so a balanced loss is approximately one regardless of the family weight.

Routing mutual information is calculated independently at each recurrent step:

```text
MI = entropy(mean expert probability across examples)
     - mean entropy(per-example expert probability)
```

The training objective subtracts `route_mi_coefficient * MI`, rewarding
confident per-example choices that remain diverse across the batch. Calculating
it per step means two examples taking opposite expert orders receive a reward,
while a batch following one shared order does not receive diversity merely
because different steps use different experts. Exit is again excluded.

The prepared full-data command is:

```bash
python -m scripts.train_mnist \
  --preset diverse_full \
  --variant straight_through \
  --seed 7 \
  --device cuda
```

This command trains on all 60,000 official training images, uses one stratified
5,000-image half of the official test split for early stopping, and evaluates
the disjoint 5,000-image held-out half after training. Give coefficient sweeps
distinct preset and experiment names; do not repeatedly evaluate the held-out
partition while tuning.

Progress bars report the raw hierarchical balance and routing mutual
information values. Compare their weighted contributions with cross-entropy
rather than comparing coefficients alone.

On the initial 24-expert CUDA calibration batch, router gradient norms were
`0.0350` for classification, `0.0881` for unweighted hierarchical balance, and
`0.0217` for unweighted routing MI. The configured coefficients yield initial
weighted norms of approximately `0.00264` for balance and `0.00217` for routing
MI, so both auxiliary signals are material without dominating classification.

Use `make_expert_pool()` to construct a heterogeneous pool without listing every
expert:

```python
from sglr.presets.mnist import make_expert_pool

experts = make_expert_pool(
    mlp_experts=32,
    attention_experts=32,
    conv_experts=32,
)
```

The builders cycle the canonical hyperparameter templates and generate stable,
unique names. The CLI shorthand `--experts-per-family 32` creates the same
balanced 96-expert pool. Noncanonical pools disable the 200,000-parameter
research guard because the purpose is explicitly to scale expert count.
They also receive a distinct name such as `pilot_96_experts`, preventing their
checkpoints from colliding with the canonical `pilot` run.
The router count and each router's output width both grow with the number of
experts, so router parameters grow quadratically even though each expert stays
small. The 24-, 48-, and 96-expert pilot models currently contain 103,933,
222,799, and 519,283 parameters, respectively.

Use `dataclasses.replace()` inside a preset function to change batch size,
weight decay, warmup fraction, split sizes, seed, workers, latent width, patch
size, router width, or routing mode. For a reusable custom preset, add a function
beside `pilot()` and register it in `MNIST_PRESETS`:

```python
def large_pilot(*, experts_per_family: int = 32) -> ExperimentConfig:
    experiment = pilot(experts_per_family=experts_per_family)
    return replace(
        experiment,
        experiment_name="large_pilot",
        model=replace(experiment.model, max_steps=12),
        training=replace(experiment.training, learning_rate=1e-3, epochs=20),
    )
```

Then add `"large_pilot": large_pilot` to `MNIST_PRESETS` and select it with
`--preset large_pilot`. Calling `experiment.validate()` in a unit test catches
invalid shapes, duplicate expert names, routing limits, and training ranges
before a long run begins.

Use a distinct `experiment_name` or output root after changing a configuration.
The runners reject reuse of a completed run directory with a different resolved
configuration.

## Train one already-chosen configuration

After hyperparameters are frozen, train or resume one ordinary run with:

```bash
python -m scripts.train_mnist \
  --preset pilot \
  --variant straight_through \
  --seed 7 \
  --device cuda
```

Unlike `scripts.run_round1`, the ordinary runner evaluates test after training.
Do not use repeated ordinary runs for validation-driven hyperparameter search.

## Regenerate route figures

Figures can be regenerated without training:

```bash
python -m scripts.analyze_mnist run \
  --run-dir runs/round1/depth05_penalty1e3/seed_7/validation \
  --manifest runs/round1/depth05_penalty1e3/seed_7/validation_manifest.json
```

The analysis directory contains 300 dpi PNGs and their CSV/JSON sources. Treat
class-conditioned routing as useful specialization only if it improves held-out
performance and exceeds shuffled-label, frozen-random-router, and hard-routing
controls. High lift from a rarely visited expert is not sufficient evidence.
