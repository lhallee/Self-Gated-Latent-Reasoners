## Self-Gated Latent Reasoners

This repository contains an MNIST-focused research prototype for Self-Gated Latent Reasoners (SGLRs): a heterogeneous expert network where each sample can take its own discrete route through MLP, convolutional, and attention experts before exiting to a classifier.

The current codebase now supports:

- batched hard top-1 grouped routing, so different batch elements can follow different expert paths in the same forward pass;
- modular training and checkpointing under `runs/`;
- frozen-backbone probe training for per-expert class prediction and image reconstruction;
- a Streamlit explorer for route replay, gate probabilities, and probe reconstructions.

The original blog post is still here for background: [How to Build Conscious AGI](https://medium.com/minds-and-molecules/how-to-build-conscious-agi-5684526f55f0).

## Repository Layout

- `sglr/`: core package with model, router, experts, training, probes, and artifact helpers.
- `scripts/train_mnist.py`: base SGLR MNIST training entry point.
- `scripts/train_probes.py`: second-stage probe training entry point.
- `scripts/smoke_test.py`: lightweight routing/model/probe smoke validation.
- `apps/streamlit_app.py`: interactive route and probe explorer.
- `sglr_mnist.py`: compatibility wrapper that forwards to `scripts/train_mnist.py`.
- `graph_utils.py` and `residence_time.ipynb`: separate graph-simulation experiments kept intact.

## Installation

Install the Python dependencies first:

```bash
py -m pip install -r requirements.txt
```

## Quick Start

Run the lightweight validation first:

```bash
py -m scripts.smoke_test
```

Train a base MNIST model:

```bash
py -m scripts.train_mnist --run-name baseline_mnist --batch-size 32 --epochs 10
```

Example with a larger effective batch through gradient accumulation:

```bash
py -m scripts.train_mnist --run-name baseline_mnist --batch-size 16 --grad-accum-steps 4 --epochs 10
```

You can still use the old entry point if you want:

```bash
py sglr_mnist.py --run-name baseline_mnist
```

Train probes on the best frozen backbone from a completed run:

```bash
py -m scripts.train_probes --base-run-dir runs/<timestamp>_baseline_mnist --batch-size 32 --epochs 5
```

Launch the interactive analysis app:

```bash
streamlit run apps/streamlit_app.py
```

## Smoke / Small Runs

For short CPU checks, use the built-in smoke mode:

```bash
py -m scripts.train_mnist --smoke-test --run-name smoke_base
py -m scripts.train_probes --base-run-dir runs/<timestamp>_smoke_base --smoke-test
```

The smoke settings intentionally reduce subset sizes, epochs, and plotting volume so you can verify the pipeline without running a full experiment.

## Output Structure

Each training run is stored under `runs/<timestamp>_<run_name>/`.

- `base/`
  - `best_model.pt`
  - `last_model.pt`
  - `model_config.json`
  - `training_config.json`
  - `training_history.json`
  - `training_summary.json`
  - `training_curves.png`
  - `digit_usage_patterns.png`
  - `digit_route_summary.json`
- `probes/`
  - `best_probes.pt`
  - `last_probes.pt`
  - `probe_config.json`
  - `probe_history.json`
  - `probe_summary.json`
  - `per_expert_probe_summary.json`

## Notes

- Routing is discrete and grouped by selected expert for efficiency, which keeps the per-sample path interpretation clean.
- Probe heads are intentionally simple linear classifiers and linear reconstructors attached after backbone training.
- If you want full MNIST sweeps, larger runs are best executed on your workstation GPU rather than through a lightweight smoke path.
