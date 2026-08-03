# Historical experiments

This directory documents SGLR work that predates the focused tiny-expert MNIST experiment. It is retained for provenance and conceptual context, not maintained as part of the current public interface.

The historical router made hard `argmax` decisions without a straight-through task gradient. Classification loss trained whichever experts happened to be selected, but did not directly teach the router which expert or exit action improved classification. Load-balancing loss was the router's only direct objective. Consequently, earlier digit-conditioned route plots demonstrate input-dependent routing, not learned useful specialization.

Archived areas are described separately:

- `probes/`: frozen-backbone classification and reconstruction probes.
- `interactive_app/`: the Streamlit route and reconstruction viewer.
- `residence_time/`: notebook experiments about graph residence times.
- `graph_simulation/`: standalone signal-propagation utilities and animations.
- `research_notes/`: speculative future directions and conceptual notes.
- `sglr_mnist.py`: the removed compatibility entry point for the original flat-vector implementation.

These workflows may require packages intentionally omitted from the core environment, including Streamlit, NetworkX, and pandas. Use the repository history or an environment captured with the original run when reproducing them.
