#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from enum import Enum
from typing import Optional, Tuple

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.util.torch_utils import get_torch_jit_decorator


class SquashType(Enum):
    CLAMP = 0
    CLAMP_RESCALE = 1
    TANH = 2
    IDENTITY = 3


def scale_ctrl(ctrl, action_lows, action_highs, squash_fn: SquashType = SquashType.CLAMP):
    if len(ctrl.shape) == 1:
        ctrl = ctrl.unsqueeze(0).unsqueeze(-1)
        # ctrl = ctrl[np.newaxis, :, np.newaxis] # TODO: does this work with gpu pytorch?
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == SquashType.CLAMP:
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == SquashType.CLAMP_RESCALE:
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == SquashType.TANH:
        ctrl = torch.tanh(ctrl)
    elif squash_fn == SquashType.IDENTITY:
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)


def select_top_rollouts(
    total_costs: torch.Tensor,
    vis_seq: Optional[torch.Tensor],
    n_problems: int,
    particles_per_problem: int,
    top_limit: int = 20,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Prepare top rollouts for visualization buffers.

    Args:
        total_costs: Raw cost tensor aggregated per rollout.
        vis_seq: Rollout state tensor for visualization selection.
        n_problems: Number of parallel problem instances.
        particles_per_problem: Expected particle count per problem.
        top_limit: Maximum number of rollouts to keep per problem.

    Returns:
        A tuple ``(top_values, top_indices, top_trajs, reshaped_costs)``. The first
        three entries are ``None`` when there are no particles to rank.
    """

    if total_costs.dim() == 0:
        total_costs = total_costs.unsqueeze(0)
    if total_costs.numel() == 0:
        reshaped = total_costs.new_zeros(n_problems, 0)
        return None, None, None, reshaped
    if total_costs.dim() == 1 or total_costs.shape[0] != n_problems:
        reshaped = total_costs.view(n_problems, -1)
    else:
        reshaped = total_costs

    if reshaped.shape[1] == 0:
        return None, None, None, reshaped

    particle_span = particles_per_problem if particles_per_problem > 0 else reshaped.shape[1]
    particle_span = reshaped.shape[1] if particle_span == 0 else particle_span
    top_k = min(top_limit, reshaped.shape[1])
    if top_k == 0:
        return None, None, None, reshaped

    top_values, top_idx = torch.topk(reshaped, top_k, dim=1)
    if vis_seq is None:
        return top_values, top_idx, None, reshaped

    offsets = (
        torch.arange(n_problems, device=top_idx.device, dtype=top_idx.dtype)
        .mul(particle_span)
        .unsqueeze(-1)
    )
    flat_indices = top_idx + offsets
    top_trajs = torch.index_select(vis_seq, 0, flat_indices.reshape(-1))
    return top_values, top_idx, top_trajs, reshaped


#######################
## STOMP Covariance  ##
#######################


@profiler.record_function("particle_opt_utils/get_stomp_cov")
def get_stomp_cov(
    horizon: int,
    d_action: int,
    tensor_args=TensorDeviceType(),
    cov_mode="acc",
    RETURN_M=False,
):
    """Computes the covariance matrix following STOMP motion planner

    Coefficients from here: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    More info here: https://github.com/ros-industrial/stomp_ros/blob/7fe40fbe6ad446459d8d4889916c64e276dbf882/stomp_core/src/utils.cpp#L36
    """
    cov, scale_tril, scaled_M = get_stomp_cov_jit(
        horizon, d_action, cov_mode, device=tensor_args.device
    )
    if RETURN_M:
        return cov, scale_tril, scaled_M
    return cov, scale_tril


@get_torch_jit_decorator()
def get_stomp_cov_jit(
    horizon: int,
    d_action: int,
    cov_mode: str = "acc",
    device: torch.device = torch.device("cuda:0"),
):
    # This function can lead to nans. There are checks to raise an error when nan occurs.
    vel_fd_array = [0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0]

    fd_array = vel_fd_array
    A = torch.zeros(
        (d_action * horizon, d_action * horizon),
        dtype=torch.float32,
        device="cpu",
    )

    if cov_mode == "vel":
        for k in range(d_action):
            for i in range(0, horizon):
                for j in range(-3, 4):
                    # print(j)
                    index = i + j
                    if index < 0:
                        index = 0
                        continue
                    if index >= horizon:
                        index = horizon - 1
                        continue
                    A[k * horizon + i, k * horizon + index] = fd_array[j + 3]
    elif cov_mode == "acc":
        for k in range(d_action):
            for i in range(0, horizon):
                for j in range(-3, 4):
                    index = i + j
                    if index < 0:
                        index = 0
                        continue
                    if index >= horizon:
                        index = horizon - 1
                        continue

                    if index >= horizon / 2:
                        A[k * horizon + i, k * horizon - index - horizon // 2 - 1] = fd_array[j + 3]
                    else:
                        A[k * horizon + i, k * horizon + index] = fd_array[j + 3]

    R = torch.matmul(A.transpose(-2, -1).clone(), A.clone())
    M = torch.inverse(R)
    scaled_M = (1 / horizon) * M / (torch.max(torch.abs(M), dim=1)[0].unsqueeze(0))
    cov = M / torch.max(torch.abs(M))

    # also compute the cholesky decomposition:
    # scale_tril = torch.zeros((d_action * horizon, d_action * horizon), **tensor_args)

    if (cov == cov.T).all() and (torch.linalg.eigvals(cov).real >= 0).all():
        scale_tril = torch.linalg.cholesky(cov)
    else:
        scale_tril = cov

    cov = cov.to(device=device)
    scale_tril = scale_tril.to(device=device)
    scaled_M = scaled_M.to(device=device)
    """
    k = 0
    act_cov_matrix = cov[k * horizon:k * horizon + horizon, k * horizon:k * horizon + horizon]
    print(act_cov_matrix.shape)
    print(torch.det(act_cov_matrix))
    local_cholesky = matrix_cholesky(act_cov_matrix)
    for k in range(d_action):

        scale_tril[k * horizon:k * horizon + horizon,k * horizon:k * horizon + horizon] = local_cholesky
    """

    return cov, scale_tril, scaled_M


########################
## Gaussian Utilities ##
########################


def gaussian_logprob(mean, cov, x, cov_type="full"):
    """
    Calculate gaussian log prob for given input batch x
    Parameters
    ----------
    mean (np.ndarray): [N x num_samples] batch of means
    cov (np.ndarray): [N x N] covariance matrix
    x  (np.ndarray): [N x num_samples] batch of sample values

    Returns
    --------
    log_prob (np.ndarray): [num_sampls] log probability of each sample
    """
    N = cov.shape[0]
    if cov_type == "diagonal":
        cov_diag = cov.diagonal()
        cov_inv = np.diag(1.0 / cov_diag)
        cov_logdet = np.sum(np.log(cov_diag))
    else:
        cov_logdet = np.log(np.linalg.det(cov))
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    mahalanobis_dist = -0.5 * np.sum((diff @ cov_inv) * diff, axis=1)
    const1 = -0.5 * N * np.log(2.0 * np.pi)
    const2 = -0.5 * cov_logdet
    log_prob = mahalanobis_dist + const1 + const2
    return log_prob


def gaussian_logprobgrad(mean, cov, x, cov_type="full"):
    if cov_type == "diagonal":
        cov_inv = np.diag(1.0 / cov.diagonal())
    else:
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    grad = diff @ cov_inv
    return grad


def gaussian_entropy(cov=None, L=None):  # , cov_type="full"):
    """
    Entropy of multivariate gaussian given either covariance
    or cholesky decomposition of covariance

    """
    if cov is not None:
        inp_device = cov.device
        cov_logdet = torch.log(torch.det(cov))
        # print(np.linalg.det(cov.cpu().numpy()))
        # print(torch.det(cov))
        N = cov.shape[0]

    else:
        inp_device = L.device
        cov_logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        N = L.shape[0]
    # if cov_type == "diagonal":
    # cov_logdet =  np.sum(np.log(cov.diagonal()))
    # else:
    # cov_logdet = np.log(np.linalg.det(cov))

    term1 = 0.5 * cov_logdet
    # pi = torch.tensor([math.pi], device=inp_device)
    # pre-calculate 1.0 + torch.log(2.0*pi) = 2.837877066
    term2 = 0.5 * N * 2.837877066

    ent = term1 + term2
    return ent.to(inp_device)


def gaussian_kl(mean0, cov0, mean1, cov1, cov_type="full"):
    """
    KL-divergence between Gaussians given mean and covariance
    KL(p||q) = E_{p}[log(p) - log(q)]

    """
    N = cov0.shape[0]
    if cov_type == "diagonal":
        cov1_diag = cov1.diagonal()
        cov1_inv = np.diag(1.0 / cov1_diag)
        cov0_logdet = np.sum(np.log(cov0.diagonal()))
        cov1_logdet = np.sum(np.log(cov1_diag))
    else:
        cov1_inv = np.linalg.inv(cov1)
        cov0_logdet = np.log(np.linalg.det(cov0))
        cov1_logdet = np.log(np.linalg.det(cov1))

    term1 = 0.5 * np.trace(cov1_inv @ cov0)
    diff = (mean1 - mean0).T
    mahalanobis_dist = 0.5 * np.sum((diff @ cov1_inv) * diff, axis=1)
    term3 = 0.5 * (-1.0 * N + cov1_logdet - cov0_logdet)
    return term1 + mahalanobis_dist + term3


# @get_torch_jit_decorator()
def cost_to_go(cost_seq, gamma_seq, only_first=False):
    # type: (Tensor, Tensor, bool) -> Tensor

    """
    Calculate (discounted) cost to go for given cost sequence
    """
    # if torch.any(gamma_seq == 0):
    #     return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    if only_first:
        cost_seq = torch.sum(cost_seq, dim=-1, keepdim=True) / gamma_seq[..., 0]
    else:
        # cost_seq = torch.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        cost_seq = torch.fliplr(
            torch.cumsum(torch.fliplr(cost_seq), dim=-1)
        )  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


def cost_to_go_np(cost_seq, gamma_seq):
    """
    Calculate (discounted) cost to go for given cost sequence
    """
    # if np.any(gamma_seq == 0):
    #     return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted reward sequence
    cost_seq = np.cumsum(cost_seq[:, ::-1], axis=-1)[
        :, ::-1
    ]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


############
##Cholesky##
############
def matrix_cholesky(A):
    L = torch.zeros_like(A)
    for i in range(A.shape[-1]):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s = s + L[i, k] * L[j, k]

            L[i, j] = torch.sqrt(A[i, i] - s) if (i == j) else (1.0 / L[j, j] * (A[i, j] - s))
    return L


# Batched Cholesky decomp
def batch_cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s = s + L[..., i, k] * L[..., j, k]

            L[..., i, j] = (
                torch.sqrt(A[..., i, i] - s)
                if (i == j)
                else (1.0 / L[..., j, j] * (A[..., i, j] - s))
            )
    return L
