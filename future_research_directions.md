# Future Research Directions for Self-Gated Latent Reasoners

## Context From This Repository

This project has two linked research threads:

1. Residence-time simulations over directed graphs, where signal persistence is studied as a function of graph size, connectedness, output nodes, and stochastic termination.
2. Self-Gated Latent Reasoners, where MNIST samples move through heterogeneous MLP, convolutional, and attention experts until a learned exit route terminates computation.

The current SGLR implementation supports batched hard top-1 grouped routing, an explicit exit route, frozen-backbone probes, per-expert classifier and reconstruction probes, route summaries, and a Streamlit route explorer.

The strongest current baseline run in `runs/20260306_233407_baseline` reached:

- 96.57% MNIST eval accuracy.
- 8.27M parameters.
- Mean eval route depth around 4.32 expert steps at epoch 10.
- Probe eval accuracy of 95.59% over visited expert states.
- Nontrivial per-digit route motifs, with some digits showing short stable paths and others showing more route diversity.

The blog post frames the broader hypothesis as signal propagation through recurrent, self-modifying circuits with learned self-termination. The research program below keeps that philosophical framing but translates it into falsifiable ML experiments.

## Highest Priority Direction: Router Credit Assignment

The current router uses hard argmax top-1 route selection. This keeps paths interpretable, but it also means downstream classification loss does not directly differentiate through the routing decision. Router heads receive direct gradients through probability regularizers, while expert and classifier weights receive task gradients through the chosen path.

This is the first thing to make explicit and test.

### Experiments

- Compare current hard top-1 routing against straight-through estimator routing.
- Compare against Gumbel-Softmax routing with temperature annealing.
- Compare against soft mixture routing during training with hard top-1 routing at inference.
- Compare against stochastic policy-gradient routing with a compute-normalized reward.
- Add top-k routing where multiple experts can transform the latent in parallel, then merge.
- Train a task-supervised router and evaluate whether route motifs become more stable, more accurate, or more compute-efficient.

### Key Metrics

- Accuracy at equal parameter count.
- Accuracy at equal FLOPs.
- Mean and distribution of route depth.
- Forced-exit rate when max steps are reached.
- Route entropy per step.
- Mutual information between route sequence and class label.
- Route stability under small input augmentations.
- Router gradient norms by loss term.

### Minimal Implementation Target

Add a `routing_mode` option with:

- `hard_argmax`
- `straight_through`
- `gumbel_softmax`
- `soft_mixture`
- `top_k`

Then run each mode under the same model size, seed set, training budget, and plotting pipeline.

## Residence Time as a Learned Quantity

The graph simulations define residence time as the number of steps a signal persists before termination. The SGLR model has a direct analogue: number of expert transitions before exit, plus the shape of the route sequence.

### Experiments

- Treat learned route depth as residence time and plot survival curves by class, confidence, correctness, and input difficulty.
- Compare random graph residence-time statistics against learned SGLR transition graphs.
- Build an empirical route-transition matrix from trained models and measure absorbing-state behavior around the exit node.
- Estimate hitting times to exit under learned routing probabilities.
- Track how route-depth distributions evolve over training.
- Measure whether long-residence samples are harder, more ambiguous, more OOD, or simply router artifacts.

### Useful Analyses

- Route-depth histograms by digit.
- Accuracy conditional on route depth.
- Confidence conditional on route depth.
- Error rate for samples that hit max steps.
- Strongly connected components in learned transition graphs.
- Spectral radius or mixing behavior of non-exit route transitions.

## Exit Policy and Adaptive Computation

The exit route is the core distinction from ordinary MoE. It should be studied as a calibrated adaptive-computation policy.

### Experiments

- Add a ponder cost that penalizes longer routes.
- Add a minimum-compute reward or target residence-time regularizer to avoid trivial immediate exits.
- Train auxiliary classifiers at every step and apply anytime losses.
- Evaluate whether early exits are calibrated by confidence.
- Force exits at different depths and measure accuracy degradation.
- Let the model abstain when no exit confidence threshold is reached.

### Baselines

- Fixed-depth expert stacks.
- Recurrent shared expert with the same compute budget.
- Standard MoE without recurrence.
- Early-exit CNN or MLP.
- Random router with learned experts.
- Learned router with frozen random experts.

## Expert Specialization and Route Semantics

The current experts are heterogeneous but not explicitly encouraged to specialize beyond task loss and load balancing. Future work should distinguish true specialization from incidental routing.

### Experiments

- Quantify expert specialization using class enrichment, reconstruction quality, input gradients, and latent neighborhood changes.
- Add expert dropout to force redundant or robust routes.
- Add expert-type priors, such as local conv experts early and global attention experts later, then compare to unconstrained routing.
- Learn expert birth, pruning, and merging from route usage statistics.
- Add residual scaling per expert so the model can learn how strongly each expert edits the latent.
- Compare homogeneous expert pools against heterogeneous pools.

### Causal Tests

- Swap one expert in a learned route with another expert and measure output changes.
- Replay a route while forcing one step to a counterfactual expert.
- Shuffle route sequences across samples.
- Freeze the router and retrain experts.
- Freeze experts and retrain only the router.
- Remove high-usage experts after training and measure degradation.

## Probe-Based Interpretability

The probe system is already a useful foundation. It can become a full interpretability suite for latent reasoning trajectories.

### Experiments

- Train probes on pre-step and post-step latents separately.
- Train nonlinear probes and compare against current linear probes.
- Measure when class information becomes linearly decodable along each route.
- Add reconstruction probes for denormalized image space and feature-space reconstructions.
- Track probe confidence changes step by step.
- Identify routes where probe confidence decreases before improving, which may indicate useful latent rewriting.
- Add probe disagreement metrics between experts and the final classifier.

### Artifacts to Add

- Per-sample route trace JSON export.
- Route-cluster reports.
- Probe trajectory plots.
- Confusion matrices by route family.
- UMAP or PCA plots of pre-step and post-step latents colored by expert, digit, route depth, and correctness.

## Robustness and OOD Behavior

Adaptive routing should be valuable if route depth and route choice respond to uncertainty.

### Experiments

- Evaluate on rotated, translated, blurred, occluded, and noisy MNIST.
- Evaluate MNIST-trained models on Fashion-MNIST or KMNIST as OOD inputs.
- Measure whether OOD samples have longer residence time, lower exit confidence, or more route entropy.
- Add adversarial perturbations and measure route instability.
- Train with augmentations and test whether routes become smoother under small transformations.

### Metrics

- OOD AUROC from route depth.
- OOD AUROC from route entropy.
- OOD AUROC from exit confidence.
- Accuracy under corruption severity.
- Route edit distance under paired clean and corrupted inputs.

## Better Benchmarks

MNIST proves the implementation works, but it is too simple to validate latent reasoning claims.

### Next Benchmarks

- Fashion-MNIST and KMNIST for quick image-domain checks.
- CIFAR-10 and CIFAR-100 for richer visual structure.
- SVHN for digit recognition with harder backgrounds.
- Sequential MNIST or permuted MNIST for recurrence-sensitive routing.
- Small algorithmic tasks where multi-step computation should matter.
- Compositional visual tasks such as CLEVR-like synthetic reasoning.
- Tiny language tasks where exit depth can track ambiguity.

### Evaluation Principle

For every benchmark, report accuracy, compute, route depth, route entropy, forced-exit rate, and parameter count. The central claim should be accuracy-compute Pareto improvement, not raw accuracy alone.

## Scaling Toward Protein and Multimodal Biology Tasks

The architecture is naturally aligned with biological multimodality: sequence, structure, annotation text, residue graphs, interaction networks, and assay context can each have specialized experts.

### Protein-Focused Expert Families

- Sequence transformer expert initialized from a protein language model.
- Lightweight convolutional motif expert for local sequence patterns.
- Structure graph neural network expert for residue contact graphs.
- Annotation text expert for GO terms, EC descriptions, and free-text metadata.
- MSA or homolog-context expert where alignments are available.
- Codon-aware nucleotide expert for gene-level tasks.
- Interaction-network expert for PPI context.
- Diffusion denoising expert for generation or sequence refinement.

### Tasks

- GO and EC prediction.
- Remote homology classification.
- Protein-protein interaction prediction.
- Mutation effect prediction from DMS.
- Subcellular localization.
- Binding-site or active-site prediction.
- Conditional protein generation.
- Sequence-to-function retrieval.

### Practical Starting Point

Use a frozen PLM backbone to produce sequence latents, then train lightweight routed LoRA or adapter experts on top. This would test whether recurrent expert routing adds value without immediately paying the cost of full PLM finetuning.

## Generative SGLRs

The blog post's signal-refinement framing maps cleanly onto iterative generation and diffusion.

### Experiments

- Use expert routes as a learned denoising schedule.
- Let the router choose between local motif repair, global consistency, structure constraint, and annotation-alignment experts.
- Train masked-token protein generation where the model exits when the sequence is sufficiently refined.
- Track residence time by generation difficulty.
- Compare fixed denoising schedules against learned self-termination.

## Multimodal Latent Workspace

The speculative "brain section" idea can be made concrete without relying on consciousness claims.

### Experiments

- Build modality-specific encoders for image, text, audio, sequence, and graph inputs.
- Route a shared latent through modality experts and a small global workspace expert.
- Train contrastive and reconstruction objectives so routes must preserve cross-modal alignment.
- Add real-time or streaming inputs and measure whether route depth adapts to changing evidence.
- Compare single-shared-router, per-modality-router, and pairwise-router designs.

### Caution

Do not evaluate this as a consciousness model. Evaluate it as adaptive multimodal computation with interpretable routing, memory, and termination.

## Training and Systems Work

Hard routed recurrence can become slow or memory-heavy as expert count and route depth grow.

### Directions

- Keep grouped dispatch, but benchmark dispatch overhead by batch size, expert count, and route depth.
- Add mixed precision and `torch.compile` compatibility checks.
- Add gradient checkpointing for long routes.
- Add distributed expert parallelism once expert count grows.
- Log per-expert FLOPs and wall-clock usage.
- Cache route traces and latents for probe training.
- Add structured experiment configs so sweeps are reproducible.

## Suggested Roadmap

### Phase 1: Make the Current Claim Sharp

- Add differentiable router variants.
- Add route-depth, route-entropy, forced-exit, and route-stability metrics.
- Run MNIST with at least 5 seeds per routing mode.
- Compare against fixed-depth and no-recurrence baselines.

### Phase 2: Show Adaptive Computation

- Add ponder cost and anytime losses.
- Run clean, corrupted, and OOD MNIST-style benchmarks.
- Show accuracy-compute Pareto curves.
- Test whether residence time predicts uncertainty or error.

### Phase 3: Show Meaningful Specialization

- Add causal route interventions.
- Expand probes to pre-step and post-step latents.
- Quantify expert specialization and route motifs.
- Add route-cluster visualizations.

### Phase 4: Leave MNIST

- Move to CIFAR, algorithmic tasks, and a small protein function task.
- Start with frozen PLM latents plus routed adapter experts.
- Compare against simple adapters, fixed-depth adapters, and standard MoE adapters.

### Phase 5: Multimodal and Generative Extensions

- Add sequence, structure, and text experts for protein tasks.
- Add diffusion-style refinement experts.
- Test learned exit policies for generation.
- Evaluate whether route depth tracks sample ambiguity, constraint difficulty, or biological novelty.

## Candidate Paper Thesis

Self-Gated Latent Reasoners provide a simple framework for adaptive recurrent computation in which samples choose interpretable expert trajectories and terminate when sufficient evidence has accumulated. The key empirical questions are whether learned routing improves accuracy-compute tradeoffs, whether route residence time measures uncertainty or difficulty, and whether heterogeneous experts acquire causal specialization.

## Immediate Next Experiments

1. Implement straight-through and Gumbel-Softmax router modes.
2. Add route metrics to `training_summary.json` and per-epoch history.
3. Add fixed-depth, random-router, and no-recurrence baselines.
4. Run 5-seed MNIST sweeps with equal parameter and equal compute comparisons.
5. Evaluate clean versus corrupted MNIST and test route depth as an uncertainty signal.
6. Add causal route intervention scripts using saved traces.
7. Prototype a frozen-PLM routed adapter model for one protein function benchmark.

