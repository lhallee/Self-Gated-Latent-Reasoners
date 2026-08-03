# Running and modifying SGLR experiments

The paused round-one results are summarized in
[`round1_partial_findings.md`](round1_partial_findings.md).

## Resume the first validation-only depth round

The first round is resumable. Completed candidates are skipped, and an interrupted
candidate resumes from its last complete epoch:

```bash
python -m scripts.run_round1 \
  --config configs/mnist/pilot.toml \
  --output-root runs/round1 \
  --device cuda \
  --epochs 20 \
  --patience 5
```

Candidate selection uses only the 2,000-example validation subset. The script
evaluates the official test set once, after all candidates finish and the winner
is frozen. If `runs/round1/selected_test/run_complete.json` exists, rerunning the
command does not evaluate test again.

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
`cuda`.

## Change the model and data preset

The TOML files under `configs/mnist/` are the source of the common model,
expert-pool, optimizer, and data settings:

- `smoke.toml`: 256 train, 128 validation, one epoch.
- `pilot.toml`: 12,000 train and 2,000 validation.
- `full.toml`: 50,000 train and 10,000 validation.
- `focused.toml`: the four-variant, three-seed comparison.

Edit `[training]` to change batch size, weight decay, warmup fraction, split
sizes, seed, worker count, or logging frequency. Edit `[model]` to change latent
width, patch size, router width, or the default routing mode. Every
`[[model.experts]]` table defines one named expert and its family-specific
width, heads, channels, kernel, or dilation.

Use a distinct `experiment_name` or output root after changing a configuration.
The runners reject reuse of a completed run directory with a different resolved
configuration.

## Train one already-chosen configuration

After hyperparameters are frozen, train or resume one ordinary run with:

```bash
python -m scripts.train_mnist \
  --config configs/mnist/pilot.toml \
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
