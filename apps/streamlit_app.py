from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglr.analysis import extract_route_sequence, flatten_images
from sglr.artifacts import load_checkpoint, load_json
from sglr.config import ModelConfig
from sglr.model import SGLRModel
from sglr.probes import ExpertProbeSuite, run_probes_on_trace
from sglr.train import build_mnist_transforms, select_device


st.set_page_config(page_title="SGLR MNIST Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def list_run_directories(output_root: str) -> list[str]:
    output_root_path = Path(output_root)
    if not output_root_path.exists():
        return []
    run_directories = [path.name for path in output_root_path.iterdir() if path.is_dir()]
    run_directories.sort(reverse=True)
    return run_directories


@st.cache_resource(show_spinner=False)
def load_run_bundle(run_directory: str, device_name: str):
    run_dir = Path(run_directory)
    base_stage_dir = run_dir / "base"
    probe_stage_dir = run_dir / "probes"
    device = select_device(device_name)
    model_config = ModelConfig(**load_json(base_stage_dir / "model_config.json"))
    model = SGLRModel(model_config).to(device)
    checkpoint = load_checkpoint(base_stage_dir / "best_model.pt", device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probe_suite = None
    probe_checkpoint_path = probe_stage_dir / "best_probes.pt"
    if probe_checkpoint_path.exists():
        probe_suite = ExpertProbeSuite(
            num_experts=model_config.num_experts,
            input_size=model_config.input_size,
            num_classes=model_config.num_classes,
        ).to(device)
        probe_checkpoint = load_checkpoint(probe_checkpoint_path, device=device)
        probe_suite.load_state_dict(probe_checkpoint["probe_state"])
        probe_suite.eval()

    return model, probe_suite, device


@st.cache_resource(show_spinner=False)
def load_test_dataset(data_root: str):
    return torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=build_mnist_transforms())


def build_filtered_indices(dataset, digit_filter: str) -> list[int]:
    if digit_filter == "All":
        return list(range(len(dataset)))
    digit_value = int(digit_filter)
    return [int(index) for index, target in enumerate(dataset.targets.tolist()) if target == digit_value]


def make_route_graph_figure(model: SGLRModel, route_sequence: tuple[int, ...]):
    graph = nx.DiGraph()
    graph.add_node("start")
    graph.add_node("exit")
    for expert_name in model.expert_names:
        graph.add_node(expert_name)
        graph.add_edge("start", expert_name)
        graph.add_edge(expert_name, "exit")

    positions = {"start": (-1.8, 0.0), "exit": (1.8, 0.0)}
    angles = np.linspace(0.0, 2.0 * np.pi, num=len(model.expert_names), endpoint=False)
    for expert_name, angle in zip(model.expert_names, angles):
        positions[expert_name] = (np.cos(angle), np.sin(angle))

    route_names = ["start"] + [model.route_name(route_id) for route_id in route_sequence]
    path_edges = list(zip(route_names[:-1], route_names[1:]))
    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(graph, positions, edgelist=list(graph.edges()), alpha=0.12, ax=axis, edge_color="gray")
    nx.draw_networkx_nodes(graph, positions, nodelist=model.expert_names, node_color="lightsteelblue", node_size=1100, ax=axis)
    nx.draw_networkx_nodes(graph, positions, nodelist=["start"], node_color="lightgreen", node_size=1200, ax=axis)
    nx.draw_networkx_nodes(graph, positions, nodelist=["exit"], node_color="khaki", node_size=1200, ax=axis)
    if path_edges:
        nx.draw_networkx_edges(graph, positions, edgelist=path_edges, edge_color="crimson", width=3.0, ax=axis, arrows=True)
        visited_nodes = list(dict.fromkeys(route_names))
        nx.draw_networkx_nodes(graph, positions, nodelist=visited_nodes, node_color="salmon", node_size=1200, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=9, ax=axis)
    axis.set_title("Expert Routing Graph")
    axis.axis("off")
    figure.tight_layout()
    return figure


def make_gate_heatmap_figure(model: SGLRModel, trace, sample_index: int):
    step_count = trace.executed_steps
    probability_matrix = trace.route_probs[:step_count, sample_index].detach().cpu().T
    route_labels = [model.route_name(route_id) for route_id in range(probability_matrix.size(0))]
    figure, axis = plt.subplots(figsize=(max(7, step_count * 0.75), 4))
    image = axis.imshow(probability_matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(step_count), labels=[str(step) for step in range(step_count)])
    axis.set_yticks(range(len(route_labels)), labels=route_labels)
    axis.set_xlabel("Routing Step")
    axis.set_ylabel("Destination")
    axis.set_title("Gate Probabilities")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def build_step_dataframe(model: SGLRModel, trace, sample_index: int, probe_output) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step_index in range(trace.executed_steps):
        if not trace.active_mask[step_index, sample_index]:
            break
        route_id = int(trace.route_ids[step_index, sample_index].item())
        route_probabilities = trace.route_probs[step_index, sample_index]
        top_probabilities, top_indices = torch.topk(route_probabilities, k=min(3, route_probabilities.numel()))
        top_routes = [
            f"{model.route_name(int(route_index.item()))}: {float(probability.item()):.3f}"
            for probability, route_index in zip(top_probabilities, top_indices)
        ]
        row = {
            "step": step_index,
            "selected_route": model.route_name(route_id),
            "top_routes": " | ".join(top_routes),
        }
        if probe_output is not None and route_id != trace.exit_route_index:
            step_logits = probe_output.classifier_logits[step_index, sample_index]
            step_probabilities = step_logits.softmax(dim=-1)
            row["probe_prediction"] = int(step_probabilities.argmax().item())
            row["probe_confidence"] = float(step_probabilities.max().item())
        else:
            row["probe_prediction"] = None
            row["probe_confidence"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def make_probe_gallery_figure(trace, probe_output, sample_index: int, route_name_fn):
    visited_steps = [
        step_index
        for step_index in range(trace.executed_steps)
        if int(trace.route_ids[step_index, sample_index].item()) != trace.exit_route_index
    ]
    if not visited_steps:
        return None

    column_count = min(4, len(visited_steps))
    row_count = int(np.ceil(len(visited_steps) / column_count))
    figure, axes = plt.subplots(row_count, column_count, figsize=(column_count * 2.5, row_count * 2.8))
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)

    for axis in axes_array.flatten():
        axis.axis("off")

    for subplot_index, step_index in enumerate(visited_steps):
        axis = axes_array.flatten()[subplot_index]
        route_id = int(trace.route_ids[step_index, sample_index].item())
        reconstruction = probe_output.reconstructions[step_index, sample_index].detach().cpu().view(28, 28)
        probe_logits = probe_output.classifier_logits[step_index, sample_index].detach().cpu()
        predicted_digit = int(probe_logits.softmax(dim=-1).argmax().item())
        axis.imshow(reconstruction, cmap="gray")
        axis.set_title(f"Step {step_index}\n{route_name_fn(route_id)} -> {predicted_digit}", fontsize=9)
        axis.axis("off")

    figure.tight_layout()
    return figure


def main() -> None:
    st.title("SGLR MNIST Explorer")
    st.caption("Inspect trained routes, latent probes, and reconstruction behavior for Self-Gated Latent Reasoners.")

    output_root = st.sidebar.text_input("Runs directory", value="runs")
    run_names = list_run_directories(output_root)
    if not run_names:
        st.warning("No run directories were found. Train a base model first.")
        return

    selected_run_name = st.sidebar.selectbox("Run", options=run_names)
    device_name = st.sidebar.selectbox("Device", options=["cpu", "auto"], index=0)
    data_root = st.sidebar.text_input("MNIST data directory", value="data")
    run_directory = Path(output_root) / selected_run_name
    model, probe_suite, device = load_run_bundle(str(run_directory), device_name)
    dataset = load_test_dataset(data_root)

    digit_filter = st.sidebar.selectbox("Digit filter", options=["All"] + [str(digit) for digit in range(10)])
    filtered_indices = build_filtered_indices(dataset, digit_filter)
    sample_position = st.sidebar.slider("Filtered sample position", min_value=0, max_value=max(0, len(filtered_indices) - 1), value=0)
    sample_index = filtered_indices[sample_position]
    image, label = dataset[sample_index]
    image_batch = flatten_images(image.unsqueeze(0)).to(device)

    with torch.no_grad():
        output = model(image_batch)
        probe_output = None if probe_suite is None else run_probes_on_trace(probe_suite, output.trace)

    probabilities = output.logits.softmax(dim=-1)[0].detach().cpu()
    predicted_label = int(probabilities.argmax().item())
    route_sequence = extract_route_sequence(output.trace, sample_index=0)
    route_names = [model.route_name(route_id) for route_id in route_sequence]

    metric_columns = st.columns(4)
    metric_columns[0].metric("True Label", str(int(label)))
    metric_columns[1].metric("Predicted Label", str(predicted_label))
    metric_columns[2].metric("Confidence", f"{float(probabilities.max().item()):.3f}")
    metric_columns[3].metric("Route Length", str(len(route_sequence)))

    sample_columns = st.columns([1, 1.5])
    with sample_columns[0]:
        st.subheader("Input")
        st.image(image.squeeze(0).numpy(), clamp=True, use_container_width=True)
        st.write("Route")
        st.code(" -> ".join(route_names))
    with sample_columns[1]:
        st.subheader("Class Probabilities")
        probability_frame = pd.DataFrame(
            {
                "digit": list(range(probabilities.numel())),
                "probability": probabilities.tolist(),
            }
        ).sort_values("probability", ascending=False)
        st.dataframe(probability_frame, use_container_width=True, hide_index=True)

    route_graph_figure = make_route_graph_figure(model, route_sequence)
    gate_heatmap_figure = make_gate_heatmap_figure(model, output.trace, sample_index=0)
    figure_columns = st.columns(2)
    with figure_columns[0]:
        st.subheader("Expert Graph")
        st.pyplot(route_graph_figure)
    with figure_columns[1]:
        st.subheader("Gate Heatmap")
        st.pyplot(gate_heatmap_figure)

    st.subheader("Per-Step Trace")
    st.dataframe(build_step_dataframe(model, output.trace, sample_index=0, probe_output=probe_output), use_container_width=True, hide_index=True)

    if probe_output is not None:
        st.subheader("Probe Reconstructions")
        probe_gallery_figure = make_probe_gallery_figure(output.trace, probe_output, sample_index=0, route_name_fn=model.route_name)
        if probe_gallery_figure is not None:
            st.pyplot(probe_gallery_figure)

    st.subheader("Saved Run Artifacts")
    artifact_columns = st.columns(2)
    training_curves_path = run_directory / "base" / "training_curves.png"
    route_plot_path = run_directory / "base" / "digit_usage_patterns.png"
    with artifact_columns[0]:
        if training_curves_path.exists():
            st.image(str(training_curves_path), caption="Training Curves", use_container_width=True)
    with artifact_columns[1]:
        if route_plot_path.exists():
            st.image(str(route_plot_path), caption="Digit Route Patterns", use_container_width=True)

    per_expert_summary_path = run_directory / "probes" / "per_expert_probe_summary.json"
    if per_expert_summary_path.exists():
        st.subheader("Per-Expert Probe Summary")
        probe_summary = load_json(per_expert_summary_path)
        st.dataframe(pd.DataFrame(probe_summary["experts"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
