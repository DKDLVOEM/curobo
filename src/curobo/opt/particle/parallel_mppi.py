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
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.opt.particle.particle_opt_base import ParticleOptBase, ParticleOptConfig, SampleMode
from curobo.opt.particle.particle_opt_utils import (
    SquashType,
    cost_to_go,
    gaussian_entropy,
    matrix_cholesky,
    scale_ctrl,
    select_top_rollouts,
)
from curobo.rollout.rollout_base import RolloutBase, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.robot import State
from curobo.util.logger import log_info
from curobo.util.sample_lib import HaltonSampleLib, SampleConfig, SampleLib
from curobo.util.tensor_util import copy_tensor
from curobo.util.torch_utils import get_torch_jit_decorator


class BaseActionType(Enum):
    REPEAT = 0
    NULL = 1
    RANDOM = 2


class CovType(Enum):
    SIGMA_I = 0
    DIAG_A = 1
    FULL_A = 2
    FULL_HA = 3


@dataclass
class ParallelMPPIConfig(ParticleOptConfig):
    init_mean: float
    init_cov: float
    base_action: BaseActionType
    step_size_mean: float
    step_size_cov: float
    null_act_frac: float
    squash_fn: SquashType
    cov_type: CovType
    sample_params: SampleConfig
    update_cov: bool
    random_mean: bool
    beta: float
    alpha: float
    gamma: float
    kappa: float
    sample_per_problem: bool
    hierarchy_cfg: Optional[Dict[str, Any]] = None  # NEW added: 계층형 임피던스 제어 구성을 보관. (한국어 주석)

    def __post_init__(self):
        self.init_cov = self.tensor_args.to_device(self.init_cov).unsqueeze(0)
        self.init_mean = self.tensor_args.to_device(self.init_mean).clone()
        if self.hierarchy_cfg is not None and "feasibility_thresholds" in self.hierarchy_cfg:
            thresholds = self.hierarchy_cfg["feasibility_thresholds"]
            if isinstance(thresholds, dict):
                # NEW added: 설정 파일의 수치들을 부동소수로 강제 변환해 사용 편의를 높임. (한국어 주석)
                self.hierarchy_cfg["feasibility_thresholds"] = {
                    k: float(v) for k, v in thresholds.items()
                }
        if self.hierarchy_cfg is not None and "task_weights" in self.hierarchy_cfg:
            weights = self.hierarchy_cfg["task_weights"]
            if isinstance(weights, dict):
                # NEW added: 작업 가중치도 float으로 변환해 연산 안정성을 확보. (한국어 주석)
                self.hierarchy_cfg["task_weights"] = {
                    k: float(v) for k, v in weights.items()
                }
        return super().__post_init__()

    @staticmethod
    @profiler.record_function("parallel_mppi_config/create_data_dict")
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict = ParticleOptConfig.create_data_dict(
            data_dict, rollout_fn, tensor_args, child_dict
        )
        child_dict["base_action"] = BaseActionType[child_dict["base_action"]]
        child_dict["squash_fn"] = SquashType[child_dict["squash_fn"]]
        child_dict["cov_type"] = CovType[child_dict["cov_type"]]
        child_dict["sample_params"]["d_action"] = rollout_fn.d_action
        child_dict["sample_params"]["horizon"] = rollout_fn.action_horizon
        child_dict["sample_params"]["tensor_args"] = tensor_args
        child_dict["sample_params"] = SampleConfig(**child_dict["sample_params"])

        # init_mean:
        if "init_mean" not in child_dict:
            child_dict["init_mean"] = rollout_fn.get_init_action_seq()
        return child_dict


class ParallelMPPI(ParticleOptBase, ParallelMPPIConfig):
    @profiler.record_function("parallel_mppi/init")
    def __init__(self, config: Optional[ParallelMPPIConfig] = None):
        if config is not None:
            ParallelMPPIConfig.__init__(self, **vars(config))
        ParticleOptBase.__init__(self)

        self.sample_lib = SampleLib(self.sample_params)
        self._sample_set = None
        self._sample_iter = None
        # initialize covariance types:
        if self.cov_type == CovType.FULL_HA:
            self.I = torch.eye(
                self.action_horizon * self.d_action,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )

        else:  # AxA
            self.I = torch.eye(
                self.d_action, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

        self.Z_seq = torch.zeros(
            1,
            self.action_horizon,
            self.d_action,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        self.delta = None
        self.mean_action = None
        self.act_seq = None
        self.cov_action = None
        self.best_traj = None
        self.scale_tril = None
        self.visual_traj = None
        if self.debug_info is not None and "visual_traj" in self.debug_info.keys():
            self.visual_traj = self.debug_info["visual_traj"]
        self.top_values = None
        self.top_idx = None
        self.top_trajs = None

        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )

        self.reset_distribution()
        self.update_samples()
        self._use_cuda_graph = False
        self._init_cuda_graph = False
        self.info = dict(rollout_time=0.0, entropy=[])
        self._batch_size = -1
        self._store_debug = False
        self.hierarchy_cfg = self.hierarchy_cfg or {}  # NEW added: 계층 구성 미지정 시 빈 dict로 초기화. (한국어 주석)
        # NEW added: 계층형 비용 가중치 구성을 읽어 MPPI에 적용하기 위한 내부 변수 초기화. (한국어 주석)
        self._hierarchy_order: List[str] = self.hierarchy_cfg.get("task_order", [])
        self._hierarchy_thresholds: Dict[str, float] = self.hierarchy_cfg.get(
            "feasibility_thresholds", {}
        )
        self._hierarchy_weights: Dict[str, float] = self.hierarchy_cfg.get("task_weights", {})
        self._hierarchy_penalty: float = float(
            self.hierarchy_cfg.get("violation_penalty", 1e4)
        )
        self._nullspace_damping: float = float(
            self.hierarchy_cfg.get("nullspace_damping", 0.0)
        )

    def get_rollouts(self):
        return self.top_trajs

    def reset_distribution(self):
        """
        Reset control distribution
        """

        self.reset_mean()
        self.reset_covariance()

    def _compute_total_cost(self, costs, task_costs=None, task_labels=None):
        """
        Calculate weights using exponential utility
        """

        if task_costs is not None and len(self._hierarchy_order) > 0:
            combined = self._combine_task_costs(task_costs, task_labels)  # NEW added: 계층 가중 비용을 반영. (한국어 주석)
            if combined is not None:
                costs = combined
        cost_seq = jit_compute_total_cost(self.gamma_seq, costs)
        return cost_seq

    def _exp_util(self, total_costs):
        w = jit_calculate_exp_util(self.beta, total_costs)
        # w = torch.softmax((-1.0 / self.beta) * total_costs, dim=-1)
        return w

    def _exp_util_from_costs(self, costs):
        w = jit_calculate_exp_util_from_costs(costs, self.gamma_seq, self.beta)
        return w

    def _compute_mean(self, w, actions):
        # get the new means from here
        new_mean = torch.sum(w * actions, dim=-3)
        new_mean = jit_blend_mean(self.mean_action, new_mean, self.step_size_mean)
        return new_mean

    def _compute_mean_covariance(self, costs, actions):
        if self.cov_type == CovType.FULL_A:
            log_error("Not implemented")
        if self.cov_type == CovType.DIAG_A:
            new_mean, new_cov, new_scale_tril = jit_mean_cov_diag_a(
                costs,
                actions,
                self.gamma_seq,
                self.mean_action,
                self.cov_action,
                self.step_size_mean,
                self.step_size_cov,
                self.kappa,
                self.beta,
            )
            self.scale_tril.copy_(new_scale_tril)
            # self._update_cov_scale(new_cov)

        else:
            w = self._exp_util_from_costs(costs)
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
            new_cov = self._compute_covariance(w, actions)
        self._update_cov_scale(new_cov)

        return new_mean, new_cov

    def _combine_task_costs(self, task_costs: torch.Tensor, task_labels: Optional[Sequence[str]]):
        # NEW added: 계층형 우선순위를 고려해 샘플 비용을 재조합하는 전처리 단계. (한국어 주석)
        if task_costs is None or task_costs.numel() == 0:
            return None
        if len(self._hierarchy_order) == 0:
            return None
        if task_labels is None:
            task_labels = self._hierarchy_order

        label_to_index = {label: idx for idx, label in enumerate(task_labels)}  # NEW added: 작업 레이블-인덱스 맵 구성. (한국어 주석)
        ordered_indices = [label_to_index.get(label) for label in self._hierarchy_order]
        ordered_indices = [idx for idx in ordered_indices if idx is not None]
        if len(ordered_indices) == 0:
            return None

        horizon = task_costs.shape[-1]
        num_tasks = task_costs.shape[-2]
        particles_per_problem = self.total_num_particles // self.n_problems
        reshaped = task_costs.view(self.n_problems, particles_per_problem, num_tasks, horizon)  # NEW added: 문제-입자-작업 구조로 재배열. (한국어 주석)
        combined = torch.zeros(  # NEW added: 우선순위 누적 비용을 저장할 텐서 초기화. (한국어 주석)
            self.n_problems,
            particles_per_problem,
            horizon,
            device=task_costs.device,
            dtype=task_costs.dtype,
        )
        mask = torch.ones_like(combined, dtype=torch.bool)  # NEW added: 우선순위 위반을 누적 마스킹. (한국어 주석)

        for label in self._hierarchy_order:
            idx = label_to_index.get(label)
            if idx is None:
                continue
            # NEW added: 각 작업의 가중치를 적용하고 허용 임계값을 넘으면 마스킹 처리. (한국어 주석)
            component = reshaped[:, :, idx, :]
            weight = self._hierarchy_weights.get(label, 1.0)
            component_weighted = weight * component * mask.to(component.dtype)  # NEW added: 우선순위 가중치를 적용한 비용. (한국어 주석)
            combined = combined + component_weighted  # NEW added: 누적 비용에 합산. (한국어 주석)
            threshold = self._hierarchy_thresholds.get(label)
            if threshold is not None:
                threshold_tensor = component.new_tensor(threshold)
                violation = component > threshold_tensor  # NEW added: 임계값 초과 여부 계산. (한국어 주석)
                excess = torch.where(violation, component - threshold_tensor, component.new_zeros(1))  # NEW added: 초과분 추정. (한국어 주석)
                combined = combined + self._hierarchy_penalty * excess  # NEW added: 페널티를 누적. (한국어 주석)
                mask = mask & (~violation)  # NEW added: 위반된 샘플은 이후 작업에서 제외. (한국어 주석)

        combined = combined.view(self.n_problems * particles_per_problem, horizon)  # NEW added: 원래 MPPI 형상으로 복원. (한국어 주석)
        return combined

    def _compute_covariance(self, w, actions):
        if not self.update_cov:
            return
        # w = w.squeeze(-1).squeeze(-1)
        # w = w[0, :]
        if self.cov_type == CovType.SIGMA_I:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            weighted_delta = w * (delta_actions**2)
            cov_update = torch.mean(
                torch.sum(torch.sum(weighted_delta, dim=-2), dim=-1), dim=-1, keepdim=True
            )

        elif self.cov_type == CovType.DIAG_A:

            cov_update = jit_diag_a_cov_update(w, actions, self.mean_action)

        elif self.cov_type == CovType.FULL_A:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            delta = delta_actions[0, ...]

            raise NotImplementedError
        elif self.cov_type == CovType.FULL_HA:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            delta = delta_actions[0, ...]

            weighted_delta = (
                torch.sqrt(w) * delta.view(delta.shape[0], delta.shape[1] * delta.shape[2]).T
            )  # .unsqueeze(-1)
            cov_update = torch.matmul(weighted_delta, weighted_delta.T)

        else:
            raise ValueError("Unidentified covariance type in update_distribution")
        cov_update = jit_blend_cov(self.cov_action, cov_update, self.step_size_cov, self.kappa)
        return cov_update

    def _update_cov_scale(self, new_cov=None):
        if new_cov is None:
            new_cov = self.cov_action
        if not self.update_cov:
            return
        if self.cov_type == CovType.SIGMA_I:
            self.scale_tril = torch.sqrt(new_cov)

        elif self.cov_type == CovType.DIAG_A:
            self.scale_tril.copy_(torch.sqrt(new_cov))

        elif self.cov_type == CovType.FULL_A:
            self.scale_tril = matrix_cholesky(new_cov)

        elif self.cov_type == CovType.FULL_HA:
            raise NotImplementedError

    @torch.no_grad()
    def _update_distribution(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions
        task_costs = getattr(trajectories, "task_costs", None)
        task_labels = getattr(trajectories, "task_labels", None)
        merged_costs = None
        if task_costs is not None and len(self._hierarchy_order) > 0:
            combined = self._combine_task_costs(task_costs, task_labels)
            if combined is not None:
                if costs.dim() == 3:
                    merged_costs = combined.view_as(costs)
                else:
                    merged_costs = combined
        costs_for_weights = merged_costs if merged_costs is not None else costs

        # Let's reshape to n_problems now:

        # first find the means before doing exponential utility:

        with profiler.record_function("mppi/get_best"):

            # Update best action
            if self.sample_mode == SampleMode.BEST:
                w = self._exp_util_from_costs(costs_for_weights)
                best_idx = torch.argmax(w, dim=-1)
                self.best_traj.copy_(actions[self.problem_col, best_idx])
        with profiler.record_function("mppi/store_rollouts"):

            if self.store_rollouts and self.visual_traj is not None:
                total_costs = self._compute_total_cost(costs, task_costs, task_labels)
                vis_seq = getattr(trajectories.state, self.visual_traj)
                top_values, top_idx, top_trajs, total_costs = select_top_rollouts(
                    total_costs,
                    vis_seq,
                    self.n_problems,
                    self.particles_per_problem,
                    top_limit=20,
                )
                self.top_values = top_values
                self.top_idx = top_idx
                if top_trajs is None:
                    self.top_trajs = None
                elif self.top_trajs is None or self.top_trajs.shape != top_trajs.shape:
                    self.top_trajs = top_trajs
                else:
                    self.top_trajs.copy_(top_trajs)

        if not self.update_cov:
            w = self._exp_util_from_costs(costs_for_weights)
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
        else:
            new_mean, new_cov = self._compute_mean_covariance(costs_for_weights, actions)
            self.cov_action.copy_(new_cov)

        self.mean_action.copy_(new_mean)

    @torch.no_grad()
    def sample_actions(self, init_act):
        delta = torch.index_select(self._sample_set, 0, self._sample_iter).squeeze(0)
        if not self.sample_params.fixed_samples:
            self._sample_iter[:] += 1
            self._sample_iter_n += 1
            if self._sample_iter_n >= self.n_iters:
                self._sample_iter_n = 0
                self._sample_iter[:] = 0
                log_info(
                    "Resetting sample iterations in particle opt base to 0, this is okay during graph capture"
                )
        scaled_delta = delta * self.full_scale_tril
        act_seq = self.mean_action.unsqueeze(-3) + scaled_delta
        cat_list = [act_seq]

        if self.neg_per_problem > 0:
            neg_action = -1.0 * self.mean_action
            neg_act_seqs = neg_action.unsqueeze(-3).expand(-1, self.neg_per_problem, -1, -1)
            cat_list.append(neg_act_seqs)
        if self.null_per_problem > 0:
            cat_list.append(
                self.null_act_seqs[: self.null_per_problem]
                .unsqueeze(0)
                .expand(self.n_problems, -1, -1, -1)
            )

        act_seq = torch.cat(
            (cat_list),
            dim=-3,
        )
        act_seq = act_seq.reshape(self.total_num_particles, self.action_horizon, self.d_action)
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        # NEW added: 임피던스 우선순위를 보존하기 위해 null-space 감쇠를 적용. (한국어 주석)
        act_seq = self._project_nullspace_actions(act_seq)

        # if not copy_tensor(act_seq, self.act_seq):
        #    self.act_seq = act_seq
        return act_seq  # self.act_seq

    def update_seed(self, init_act):
        self.update_init_mean(init_act)

    def update_init_mean(self, init_mean):
        # update mean:
        # init_mean = init_mean.clone()
        if init_mean.shape[0] != self.n_problems:
            init_mean = init_mean.expand(self.n_problems, -1, -1)
        if not copy_tensor(init_mean, self.mean_action):
            self.mean_action = init_mean.clone()
        if not copy_tensor(init_mean, self.best_traj):
            self.best_traj = init_mean.clone()

    def reset_mean(self):
        with profiler.record_function("mppi/reset_mean"):
            if self.random_mean:
                mean = self.mean_lib.get_samples([self.n_problems])
                self.update_init_mean(mean)
            else:
                self.update_init_mean(self.init_mean)

    def reset_covariance(self):
        with profiler.record_function("mppi/reset_cov"):
            # init_cov can either be a single value, or n_problems x 1 or n_problems x d_action

            if self.cov_type == CovType.SIGMA_I:
                # init_cov can either be a single value, or n_problems x 1
                self.cov_action = self.init_cov
                if self.init_cov.shape[0] != self.n_problems:
                    self.cov_action = self.init_cov.unsqueeze(0).expand(self.n_problems, -1)
                self.inv_cov_action = 1.0 / self.cov_action
                a = torch.sqrt(self.cov_action)
                if not copy_tensor(a, self.scale_tril):
                    self.scale_tril = a

            elif self.cov_type == CovType.DIAG_A:
                # init_cov can either be a single value, or n_problems x 1 or n_problems x 7
                init_cov = self.init_cov.clone()

                # if(init_cov.shape[-1] != self.d_action):
                if len(init_cov.shape) == 1:
                    init_cov = init_cov.unsqueeze(-1).expand(-1, self.d_action)
                if len(init_cov.shape) == 2 and init_cov.shape[-1] != self.d_action:
                    init_cov = init_cov.expand(-1, self.d_action)
                init_cov = init_cov.unsqueeze(1)
                if init_cov.shape[0] != self.n_problems:
                    init_cov = init_cov.expand(self.n_problems, -1, -1)
                if not copy_tensor(init_cov.clone(), self.cov_action):
                    self.cov_action = init_cov.clone()
                self.inv_cov_action = 1.0 / self.cov_action
                a = torch.sqrt(self.cov_action)
                if not copy_tensor(a, self.scale_tril):
                    self.scale_tril = a

            else:
                raise ValueError("Unidentified covariance type in update_distribution")

    def _get_action_seq(self, mode: SampleMode):
        if mode == SampleMode.MEAN:
            act_seq = self.mean_action  # .clone()  # [self.mean_idx]#.clone()
        elif mode == SampleMode.SAMPLE:
            delta = self.generate_noise(
                shape=torch.Size((1, self.action_horizon)),
                base_seed=self.seed + 123 * self.num_steps,
            )
            act_seq = self.mean_action + torch.matmul(delta, self.full_scale_tril)
        elif mode == SampleMode.BEST:
            act_seq = self.best_traj  # [self.mean_idx]
        else:
            raise ValueError("Unidentified sampling mode in get_next_action")

        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq

    def generate_noise(self, shape, base_seed=None):
        """
        Generate correlated noisy samples using autoregressive process
        """
        delta = self.sample_lib.get_samples(sample_shape=shape, seed=base_seed)
        return delta

    def _project_nullspace_actions(self, act_seq: torch.Tensor) -> torch.Tensor:
        # NEW added: 접촉 시 여유 자유도를 감쇠해 임피던스 우선순위를 유지. (한국어 주석)
        if self._nullspace_damping <= 0.0:
            return act_seq
        rollout = getattr(self, "rollout_fn", None)
        if rollout is None or not hasattr(rollout, "_contact_wrench"):
            return act_seq
        contact_wrench = rollout._contact_wrench
        if contact_wrench is None:
            return act_seq
        activation_norm = float(self.hierarchy_cfg.get("contact_activation_norm", 0.0))  # NEW added: 감쇠 활성화 기준 노름. (한국어 주석)
        if activation_norm > 0.0:
            wrench_norm = torch.linalg.norm(contact_wrench)
            if wrench_norm < activation_norm:
                return act_seq
        bound_cost = getattr(rollout, "bound_cost", None)  # NEW added: null-space 가중치를 보유한 비용 찾기. (한국어 주석)
        if bound_cost is None or not hasattr(bound_cost, "null_space_weight"):
            return act_seq
        weights = bound_cost.null_space_weight  # NEW added: null-space 스칼라를 불러와 감쇠 계산. (한국어 주석)
        if weights is None:
            return act_seq
        scale = 1.0 - self._nullspace_damping * weights.view(1, 1, -1)  # NEW added: 감쇠된 스케일 팩터 계산. (한국어 주석)
        scale = torch.clamp(scale, min=0.0)
        return act_seq * scale  # NEW added: 감쇠된 null-space 동작을 반환. (한국어 주석)

    @property
    def full_scale_tril(self):
        """Returns the full scale tril

        Returns:
            Tensor: dimension is (d_action, d_action)
        """
        if self.cov_type == CovType.SIGMA_I:
            return (
                self.scale_tril.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)
            )
        elif self.cov_type == CovType.DIAG_A:
            return self.scale_tril.unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)  # .cl
        elif self.cov_type == CovType.FULL_A:
            return self.scale_tril
        elif self.cov_type == CovType.FULL_HA:
            return self.scale_tril

    def _calc_val(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions
        delta = actions - self.mean_action.unsqueeze(0)

        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs

        val = -self.beta * torch.logsumexp((-1.0 / self.beta) * total_costs)
        return val

    def reset(self):
        self.reset_distribution()

        self._sample_iter[:] = 0
        self._sample_iter_n = 0
        self.update_samples()  # this helps in restarting optimization
        super().reset()

    @property
    def squashed_mean(self):
        return scale_ctrl(
            self.mean_action, self.action_lows, self.action_highs, squash_fn=self.squash_fn
        )

    @property
    def full_cov(self):
        if self.cov_type == CovType.SIGMA_I:
            return self.cov_action * self.I
        elif self.cov_type == CovType.DIAG_A:
            return torch.diag(self.cov_action)
        elif self.cov_type == CovType.FULL_A:
            return self.cov_action
        elif self.cov_type == CovType.FULL_HA:
            return self.cov_action

    @property
    def full_inv_cov(self):
        if self.cov_type == CovType.SIGMA_I:
            return self.inv_cov_action * self.I
        elif self.cov_type == CovType.DIAG_A:
            return torch.diag_embed(self.inv_cov_action)
        elif self.cov_type == CovType.FULL_A:
            return self.inv_cov_action
        elif self.cov_type == CovType.FULL_HA:
            return self.inv_cov_action

    @property
    def full_scale_tril(self):
        if self.cov_type == CovType.SIGMA_I:
            return (
                self.scale_tril.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)
            )  # .cl
        elif self.cov_type == CovType.DIAG_A:
            return self.scale_tril.unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)  # .cl
        elif self.cov_type == CovType.FULL_A:
            return self.scale_tril
        elif self.cov_type == CovType.FULL_HA:
            return self.scale_tril

    @property
    def entropy(self):
        ent_L = gaussian_entropy(L=self.full_scale_tril)
        return ent_L

    def reset_seed(self):
        self.sample_lib = SampleLib(self.sample_params)
        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )
        # resample if not fixed samples:
        self.update_samples()
        super().reset_seed()

    def update_samples(self):
        with profiler.record_function("mppi/update_samples"):
            if self.sample_params.fixed_samples:
                n_iters = 1
            else:
                n_iters = self.n_iters
            if self.sample_per_problem:
                s_set = (
                    self.sample_lib.get_samples(
                        sample_shape=[
                            self.sampled_particles_per_problem * self.n_problems * n_iters
                        ],
                        base_seed=self.seed,
                    )
                    .view(
                        n_iters,
                        self.n_problems,
                        self.sampled_particles_per_problem,
                        self.action_horizon,
                        self.d_action,
                    )
                    .clone()
                )
            else:
                s_set = self.sample_lib.get_samples(
                    sample_shape=[n_iters * (self.sampled_particles_per_problem)],
                    base_seed=self.seed,
                )
                s_set = s_set.view(
                    n_iters,
                    1,
                    self.sampled_particles_per_problem,
                    self.action_horizon,
                    self.d_action,
                )
                s_set = s_set.repeat(1, self.n_problems, 1, 1, 1).clone()
            s_set[:, :, -1, :, :] = 0.0
            if not copy_tensor(s_set, self._sample_set):
                log_info("ParallelMPPI: Updating sample set")
                self._sample_set = s_set
            if self._sample_iter is None:
                log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)

                self._sample_iter = torch.zeros(
                    (1), dtype=torch.long, device=self.tensor_args.device
                )
            else:
                self._sample_iter[:] = 0

            # if not copy_tensor(sample_iter, self._sample_iter):
            #    log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)
            #    self._sample_iter = sample_iter
            self._sample_iter_n = 0

    @torch.no_grad()
    def generate_rollouts(self, init_act=None):
        """
        Samples a batch of actions, rolls out trajectories for each particle
        and returns the resulting observations, costs,
        actions

        Parameters
        ----------
        state : dict or np.ndarray
            Initial state to set the simulation problem to
        """

        return super().generate_rollouts(init_act)


@get_torch_jit_decorator()
def jit_calculate_exp_util(beta: float, total_costs):
    w = torch.softmax((-1.0 / beta) * total_costs, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta: float):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    w = torch.softmax((-1.0 / beta) * cost_seq, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_compute_total_cost(gamma_seq, costs):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    return cost_seq


@get_torch_jit_decorator()
def jit_diag_a_cov_update(w, actions, mean_action):
    delta_actions = actions - mean_action.unsqueeze(-3)

    weighted_delta = w * (delta_actions**2)
    # weighted_delta =
    # sum across horizon and mean across particles:
    # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T  , dim=0), dim=0))
    cov_update = torch.mean(torch.sum(weighted_delta, dim=-3), dim=-2).unsqueeze(-2)
    return cov_update


@get_torch_jit_decorator()
def jit_blend_cov(cov_action, cov_update, step_size_cov: float, kappa: float):
    new_cov = (1.0 - step_size_cov) * cov_action + step_size_cov * cov_update + kappa
    return new_cov


@get_torch_jit_decorator()
def jit_blend_mean(mean_action, new_mean, step_size_mean: float):
    mean_update = (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean
    return mean_update


@get_torch_jit_decorator()
def jit_mean_cov_diag_a(
    costs,
    actions,
    gamma_seq,
    mean_action,
    cov_action,
    step_size_mean: float,
    step_size_cov: float,
    kappa: float,
    beta: float,
):
    w = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)
    w = w.unsqueeze(-1).unsqueeze(-1)
    new_mean = torch.sum(w * actions, dim=-3)
    new_mean = jit_blend_mean(mean_action, new_mean, step_size_mean)
    cov_update = jit_diag_a_cov_update(w, actions, mean_action)
    new_cov = jit_blend_cov(cov_action, cov_update, step_size_cov, kappa)
    new_tril = torch.sqrt(new_cov)
    return new_mean, new_cov, new_tril
