import torch

from curobo.opt.particle.particle_opt_utils import select_top_rollouts


def test_select_top_rollouts_with_flat_costs():
    costs = torch.arange(6.0)
    vis = torch.arange(6).unsqueeze(-1)

    top_values, top_idx, top_trajs, reshaped = select_top_rollouts(
        costs,
        vis,
        n_problems=2,
        particles_per_problem=3,
        top_limit=2,
    )

    assert reshaped.shape == (2, 3)
    assert top_values.shape == (2, 2)
    assert top_idx.shape == (2, 2)
    assert top_trajs.shape == (4, 1)
    # ensure indices respect per-problem offsets
    assert torch.equal(top_trajs[:2, 0], torch.tensor([5, 4]))
    assert torch.equal(top_trajs[2:, 0], torch.tensor([2, 1]))


def test_select_top_rollouts_without_vis_seq():
    costs = torch.tensor([[1.0, 2.0, 3.0]])

    top_values, top_idx, top_trajs, reshaped = select_top_rollouts(
        costs,
        vis_seq=None,
        n_problems=1,
        particles_per_problem=3,
        top_limit=5,
    )

    assert reshaped.shape == (1, 3)
    assert top_trajs is None
    assert torch.equal(top_values, torch.tensor([[3.0, 2.0, 1.0]]))
    assert torch.equal(top_idx, torch.tensor([[2, 1, 0]]))


def test_select_top_rollouts_with_empty_costs():
    costs = torch.tensor([])
    vis = torch.tensor([])

    top_values, top_idx, top_trajs, reshaped = select_top_rollouts(
        costs,
        vis,
        n_problems=1,
        particles_per_problem=0,
        top_limit=3,
    )

    assert top_values is None
    assert top_idx is None
    assert top_trajs is None
    assert reshaped.shape == (1, 0)
