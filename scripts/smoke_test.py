from __future__ import annotations

import torch
import torch.nn as nn

from sglr.config import ModelConfig
from sglr.model import SGLRModel
from sglr.probes import ExpertProbeSuite, compute_probe_losses, run_probes_on_trace
from sglr.router import GroupedTop1Router


class AddConstantExpert(nn.Module):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.constant


class FixedRouteHead(nn.Module):
    def __init__(self, route_id: int, num_routes: int) -> None:
        super().__init__()
        self.route_id = route_id
        self.num_routes = num_routes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.size(0), self.num_routes), fill_value=-1000.0, device=x.device, dtype=x.dtype)
        logits[:, self.route_id] = 1000.0
        return logits


def validate_grouped_router() -> None:
    batch = torch.zeros(4, 6)
    router = GroupedTop1Router(num_experts=2, max_steps=2, min_steps_before_exit=1)
    experts = nn.ModuleList([AddConstantExpert(1.0), AddConstantExpert(2.0)])
    route_heads = nn.ModuleList([FixedRouteHead(route_id=2, num_routes=3), FixedRouteHead(route_id=2, num_routes=3)])
    initial_logits = torch.tensor(
        [
            [10.0, 0.0, -10.0],
            [0.0, 10.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 10.0, -10.0],
        ]
    )
    trace = router(latents=batch, initial_logits=initial_logits, experts=experts, route_heads=route_heads)
    step_zero_routes = trace.route_ids[0].tolist()
    assert step_zero_routes == [0, 1, 0, 1], "Grouped routing should let different samples take different experts in the same step"
    assert torch.allclose(trace.final_latent[0], torch.ones(6)), "Expert 0 should update its assigned samples"
    assert torch.allclose(trace.final_latent[1], torch.full((6,), 2.0)), "Expert 1 should update its assigned samples"


def validate_model_and_probes() -> None:
    torch.manual_seed(7)
    model = SGLRModel(ModelConfig())
    inputs = torch.randn(8, 784)
    labels = torch.randint(low=0, high=10, size=(8,))
    output = model(inputs)
    assert output.logits.shape == (8, 10), "Model logits should preserve batch size"
    assert output.trace.executed_steps > 0, "Model should execute at least one routing step"
    probe_suite = ExpertProbeSuite(num_experts=model.config.num_experts, input_size=model.config.input_size, num_classes=model.config.num_classes)
    probe_output = run_probes_on_trace(probe_suite, output.trace)
    classifier_loss, reconstruction_loss, _, _ = compute_probe_losses(probe_output=probe_output, images=inputs, labels=labels)
    total_loss = classifier_loss + reconstruction_loss
    total_loss.backward()


def main() -> None:
    validate_grouped_router()
    validate_model_and_probes()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
