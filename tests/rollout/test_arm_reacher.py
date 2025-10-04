import torch

from curobo.rollout.arm_reacher import cat_sum_horizon_reacher, cat_sum_reacher


def test_cat_sum_horizon_empty_tensor_returns_zero_vector():
    empty_tensor = torch.zeros((3, 0), dtype=torch.float32)

    result = cat_sum_horizon_reacher([empty_tensor])

    assert result.shape == (3,)
    assert result.dtype == empty_tensor.dtype
    assert result.device == empty_tensor.device
    assert torch.all(result == 0)


def test_cat_sum_reacher_ignores_mismatched_shapes():
    reference = torch.ones((2, 4), dtype=torch.float32)
    mismatched = torch.ones((), dtype=torch.float32)

    result = cat_sum_reacher([reference, mismatched])

    assert torch.allclose(result, reference)
