import torch

from curobo.opt.particle.parallel_mppi import ParallelMPPI


def _make_dummy_mppi():
    obj = ParallelMPPI.__new__(ParallelMPPI)
    obj._hierarchy_order = ["safety", "tracking"]
    obj._hierarchy_thresholds = {"safety": 0.5}
    obj._hierarchy_weights = {"safety": 1.0, "tracking": 2.0}
    obj._hierarchy_penalty = 10.0
    obj._nullspace_damping = 0.0
    obj._hierarchy_threshold_tensors = {"safety": torch.tensor(0.5)}
    obj._zero_scalar = torch.tensor(0.0)
    obj.n_problems = 1
    obj.total_num_particles = 4
    obj.tensor_args = type("tensor_args", (), {"device": torch.device("cpu"), "dtype": torch.float32})()
    return obj


def test_combine_task_costs_masks_and_penalizes():
    mppi = _make_dummy_mppi()
    horizon = 3
    particles = 4
    num_tasks = 2
    task_costs = torch.tensor(
        [
            [  # particle 0
                [0.2, 0.4, 0.6],  # safety below threshold on first two steps
                [1.0, 1.0, 1.0],  # tracking should contribute everywhere mask True
            ],
            [  # particle 1
                [0.7, 0.8, 0.9],  # safety violation triggers mask
                [0.5, 0.5, 0.5],  # tracking ignored where mask False
            ],
            [  # particle 2
                [0.1, 0.3, 0.2],
                [0.2, 0.4, 0.6],
            ],
            [  # particle 3
                [0.5, 0.2, 0.1],  # exactly threshold on first step
                [0.3, 0.3, 0.3],
            ],
        ],
        dtype=torch.float32,
    )
    task_costs = task_costs.view(particles, num_tasks, horizon)

    expected = torch.zeros(particles, horizon)
    safety = task_costs[:, 0, :]
    tracking = task_costs[:, 1, :]

    penalty = torch.relu(safety - 0.5) * 10.0
    expected += safety + penalty

    mask = safety <= 0.5
    expected += 2.0 * tracking * mask

    combined = mppi._combine_task_costs(task_costs, ["safety", "tracking"])
    assert torch.allclose(combined, expected)

    # Call again to ensure cached tensors remain valid and reusable
    combined_second = mppi._combine_task_costs(task_costs, ["safety", "tracking"])
    assert torch.allclose(combined_second, expected)
