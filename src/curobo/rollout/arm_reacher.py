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
from dataclasses import dataclass
from typing import Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.rollout.cost.cost_base import CostConfig
from curobo.rollout.cost.admittance_cost import (  # NEW added: 임피던스 비용 모듈을 가져와 접촉 제어에 활용. (한국어 주석)
    AdmittanceCost,
    AdmittanceCostConfig,
)
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseCostMetric
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.cost.zero_cost import ZeroCost
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.tensor import T_BValue_float, T_BValue_int
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_max
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .arm_base import ArmBase, ArmBaseConfig, ArmCostConfig


@dataclass
class ArmReacherMetrics(RolloutMetrics):
    cspace_error: Optional[T_BValue_float] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    pose_error: Optional[T_BValue_float] = None
    goalset_index: Optional[T_BValue_int] = None
    null_space_error: Optional[T_BValue_float] = None
    admittance_error: Optional[T_BValue_float] = None  # NEW added: 임피던스 추적 오차 저장. (한국어 주석)

    def __getitem__(self, idx):
        d_list = [
            self.cost,
            self.constraint,
            self.feasible,
            self.state,
            self.cspace_error,
            self.position_error,
            self.rotation_error,
            self.pose_error,
            self.goalset_index,
            self.null_space_error,
            self.admittance_error,  # NEW added: 임피던스 오차를 메트릭 리스트에 포함. (한국어 주석)
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return ArmReacherMetrics(*idx_vals)

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return ArmReacherMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
            cspace_error=None if self.cspace_error is None else self.cspace_error.clone(),
            position_error=None if self.position_error is None else self.position_error.clone(),
            rotation_error=None if self.rotation_error is None else self.rotation_error.clone(),
            pose_error=None if self.pose_error is None else self.pose_error.clone(),
            goalset_index=None if self.goalset_index is None else self.goalset_index.clone(),
            null_space_error=(
                None if self.null_space_error is None else self.null_space_error.clone()
            ),
            admittance_error=(
                None if self.admittance_error is None else self.admittance_error.clone()
            ),
            # NEW added: 임피던스 오차 텐서를 깊은 복사해 외부 사용 시 안전성을 확보. (한국어 주석)
        )


@dataclass
class ArmReacherCostConfig(ArmCostConfig):
    pose_cfg: Optional[PoseCostConfig] = None
    cspace_cfg: Optional[DistCostConfig] = None
    straight_line_cfg: Optional[CostConfig] = None
    zero_acc_cfg: Optional[CostConfig] = None
    zero_vel_cfg: Optional[CostConfig] = None
    zero_jerk_cfg: Optional[CostConfig] = None
    link_pose_cfg: Optional[PoseCostConfig] = None
    admittance_cfg: Optional[AdmittanceCostConfig] = None  # NEW added: 임피던스 비용 설정 항목. (한국어 주석)

    @staticmethod
    def _get_base_keys():
        base_k = ArmCostConfig._get_base_keys()
        # add new cost terms:
        # NEW added: 임피던스 관련 키를 포함해 확장된 비용 사전을 구성. (한국어 주석)
        new_k = {
            "pose_cfg": PoseCostConfig,
            "cspace_cfg": DistCostConfig,
            "straight_line_cfg": CostConfig,
            "zero_acc_cfg": CostConfig,
            "zero_vel_cfg": CostConfig,
            "zero_jerk_cfg": CostConfig,
            "link_pose_cfg": PoseCostConfig,
            "admittance_cfg": AdmittanceCostConfig,
        }
        new_k.update(base_k)
        return new_k

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        k_list = ArmReacherCostConfig._get_base_keys()
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        return ArmReacherCostConfig(**data)


@dataclass
class ArmReacherConfig(ArmBaseConfig):
    cost_cfg: ArmReacherCostConfig
    constraint_cfg: ArmReacherCostConfig
    convergence_cfg: ArmReacherCostConfig

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return ArmReacherCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )


@get_torch_jit_decorator()
def _compute_g_dist_jit(rot_err_norm, goal_dist):
    # goal_cost = goal_cost.view(cost.shape)
    # rot_err_norm = rot_err_norm.view(cost.shape)
    # goal_dist = goal_dist.view(cost.shape)
    g_dist = goal_dist.unsqueeze(-1) + 10.0 * rot_err_norm.unsqueeze(-1)
    return g_dist


class ArmReacher(ArmBase, ArmReacherConfig):
    """
    .. inheritance-diagram:: curobo.rollout.arm_reacher.ArmReacher
    """

    @profiler.record_function("arm_reacher/init")
    def __init__(self, config: Optional[ArmReacherConfig] = None):
        if config is not None:
            ArmReacherConfig.__init__(self, **vars(config))
        ArmBase.__init__(self)

        # self.goal_state = None
        # self.goal_ee_pos = None
        # self.goal_ee_rot = None
        # self.goal_ee_quat = None
        self._compute_g_dist = False
        self._n_goalset = 1

        if self.cost_cfg.cspace_cfg is not None:
            self.cost_cfg.cspace_cfg.dof = self.d_action
            # self.cost_cfg.cspace_cfg.update_vec_weight(self.dynamics_model.cspace_distance_weight)
            self.dist_cost = DistCost(self.cost_cfg.cspace_cfg)
        if self.cost_cfg.pose_cfg is not None:
            self.cost_cfg.pose_cfg.waypoint_horizon = self.horizon
            self.goal_cost = PoseCost(self.cost_cfg.pose_cfg)
            if self.cost_cfg.link_pose_cfg is None:
                log_info(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.cost_cfg.link_pose_cfg = self.cost_cfg.pose_cfg
        self._link_pose_costs = {}

        if self.cost_cfg.link_pose_cfg is not None:
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_costs[i] = PoseCost(self.cost_cfg.link_pose_cfg)
        if self.cost_cfg.straight_line_cfg is not None:
            self.straight_line_cost = StraightLineCost(self.cost_cfg.straight_line_cfg)
        if self.cost_cfg.zero_vel_cfg is not None:
            self.zero_vel_cost = ZeroCost(self.cost_cfg.zero_vel_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_vel_cost.hinge_value is not None:
                self._compute_g_dist = True
        if self.cost_cfg.zero_acc_cfg is not None:
            self.zero_acc_cost = ZeroCost(self.cost_cfg.zero_acc_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_acc_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.zero_jerk_cfg is not None:
            self.zero_jerk_cost = ZeroCost(self.cost_cfg.zero_jerk_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_jerk_cost.hinge_value is not None:
                self._compute_g_dist = True

        self.admittance_cost = None  # NEW added: 임피던스 비용 인스턴스 포인터 초기화. (한국어 주석)
        if self.cost_cfg.admittance_cfg is not None:
            # NEW added: 설정이 존재하면 임피던스 비용 인스턴스를 생성한다. (한국어 주석)
            self.admittance_cost = AdmittanceCost(self.cost_cfg.admittance_cfg)

        self.z_tensor = torch.tensor(
            0, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_pose_convergence = {}

        if self.convergence_cfg.pose_cfg is not None:
            self.pose_convergence = PoseCost(self.convergence_cfg.pose_cfg)
            if self.convergence_cfg.link_pose_cfg is None:
                log_warn(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.convergence_cfg.link_pose_cfg = self.convergence_cfg.pose_cfg

        if self.convergence_cfg.link_pose_cfg is not None:
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_convergence[i] = PoseCost(self.convergence_cfg.link_pose_cfg)
        if self.convergence_cfg.cspace_cfg is not None:
            self.convergence_cfg.cspace_cfg.dof = self.d_action
            self.cspace_convergence = DistCost(self.convergence_cfg.cspace_cfg)

        # check if g_dist is required in any of the cost terms:
        self.update_params(Goal(current_state=self._start_state))
        self._contact_wrench = None  # NEW added: 측정된 렌치를 저장하는 캐시 초기화. (한국어 주석)
        self._contact_cartesian_error = None  # NEW added: EE 오차 벡터 캐시 초기화. (한국어 주석)
        self._task_cost_labels = ["safety", "admittance", "tracking"]  # NEW added: 계층형 비용 레이블 정의. (한국어 주석)

    def cost_fn(self, state: KinematicModelState, action_batch=None):
        """Compute weighted cost terms for all tasks."""

        state_batch = state.state_seq
        with profiler.record_function("cost/base"):
            base_costs = super(ArmReacher, self).cost_fn(state, action_batch, return_list=True)

        cost_terms = list(base_costs)
        safety_cost = None
        zeros_like_cost = None
        if len(cost_terms) > 0:
            safety_cost = cat_sum_reacher(cost_terms)
            zeros_like_cost = torch.zeros_like(safety_cost)
        if zeros_like_cost is None:
            zeros_like_cost = torch.zeros(
                state_batch.position.shape[0],
                state_batch.position.shape[1],
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )

        tracking_terms: List[torch.Tensor] = []
        ee_pos_batch, ee_quat_batch = state.ee_pos_seq, state.ee_quat_seq
        g_dist = None
        with profiler.record_function("cost/pose"):
            if (
                self._goal_buffer.goal_pose.position is not None
                and self.cost_cfg.pose_cfg is not None
                and self.goal_cost.enabled
            ):
                if self._compute_g_dist:
                    goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                        ee_pos_batch,
                        ee_quat_batch,
                        self._goal_buffer,
                    )
                    g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                else:
                    goal_cost = self.goal_cost.forward(
                        ee_pos_batch, ee_quat_batch, self._goal_buffer
                    )
                cost_terms.append(goal_cost)
                tracking_terms.append(goal_cost)

        with profiler.record_function("cost/link_poses"):
            if self._goal_buffer.links_goal_pose is not None and self.cost_cfg.pose_cfg is not None:
                link_poses = state.link_pose
                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        current_fn = self._link_pose_costs[k]
                        if current_fn.enabled:
                            current_pose = link_poses[k].contiguous()
                            current_pos = current_pose.position
                            current_quat = current_pose.quaternion
                            c = current_fn.forward(current_pos, current_quat, self._goal_buffer, k)
                            cost_terms.append(c)
                            tracking_terms.append(c)

        if (
            self._goal_buffer.goal_state is not None
            and self.cost_cfg.cspace_cfg is not None
            and self.dist_cost.enabled
        ):
            joint_cost = self.dist_cost.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state_batch.position,
                self._goal_buffer.batch_goal_state_idx,
            )
            cost_terms.append(joint_cost)
            tracking_terms.append(joint_cost)

        if self.cost_cfg.straight_line_cfg is not None and self.straight_line_cost.enabled:
            st_cost = self.straight_line_cost.forward(ee_pos_batch)
            cost_terms.append(st_cost)
            tracking_terms.append(st_cost)

        if self.cost_cfg.zero_acc_cfg is not None and self.zero_acc_cost.enabled:
            z_acc = self.zero_acc_cost.forward(state_batch.acceleration, g_dist)
            cost_terms.append(z_acc)
            tracking_terms.append(z_acc)

        if self.cost_cfg.zero_jerk_cfg is not None and self.zero_jerk_cost.enabled:
            z_jerk = self.zero_jerk_cost.forward(state_batch.jerk, g_dist)
            cost_terms.append(z_jerk)
            tracking_terms.append(z_jerk)

        if self.cost_cfg.zero_vel_cfg is not None and self.zero_vel_cost.enabled:
            z_vel = self.zero_vel_cost.forward(state_batch.velocity, g_dist)
            cost_terms.append(z_vel)
            tracking_terms.append(z_vel)

        admittance_cost = None  # NEW added: 임피던스 비용 초기화. (한국어 주석)
        if self.admittance_cost is not None and self.admittance_cost.enabled:
            # NEW added: 측정된 접촉 렌치와 카티시안 오차를 이용해 임피던스 비용을 평가. (한국어 주석)
            admittance_cost = self.admittance_cost.forward(
                self._contact_wrench,
                self._contact_cartesian_error,
            )
            if torch.is_tensor(admittance_cost):
                cost_terms.append(admittance_cost)
            else:
                admittance_cost = None

        with profiler.record_function("cat_sum"):
            if self.sum_horizon:
                total_cost = cat_sum_horizon_reacher(cost_terms)
            else:
                total_cost = cat_sum_reacher(cost_terms)

        tracking_cost = (
            cat_sum_reacher(tracking_terms) if len(tracking_terms) > 0 else zeros_like_cost
        )
        adm_cost = admittance_cost if admittance_cost is not None else zeros_like_cost
        safety_term = safety_cost if safety_cost is not None else zeros_like_cost
        # NEW added: 안전/임피던스/추종 비용을 계층형 MPPI가 사용하도록 저장. (한국어 주석)
        self._task_cost_buffer = torch.stack([safety_term, adm_cost, tracking_cost], dim=1)

        return total_cost

    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[ArmReacherMetrics] = None
    ) -> ArmReacherMetrics:
        if out_metrics is None:
            out_metrics = ArmReacherMetrics()
        if not isinstance(out_metrics, ArmReacherMetrics):
            out_metrics = ArmReacherMetrics(**vars(out_metrics))
        out_metrics = super(ArmReacher, self).convergence_fn(state, out_metrics)

        # compute error with pose?
        if (
            self._goal_buffer.goal_pose.position is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            (
                out_metrics.pose_error,
                out_metrics.rotation_error,
                out_metrics.position_error,
            ) = self.pose_convergence.forward_out_distance(
                state.ee_pos_seq, state.ee_quat_seq, self._goal_buffer
            )
            out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
        if (
            self._goal_buffer.links_goal_pose is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            pose_error = [out_metrics.pose_error]
            position_error = [out_metrics.position_error]
            quat_error = [out_metrics.rotation_error]
            link_poses = state.link_pose

            for k in self._goal_buffer.links_goal_pose.keys():
                if k != self.kinematics.ee_link:
                    current_fn = self._link_pose_convergence[k]
                    if current_fn.enabled:
                        # get link pose
                        current_pos = link_poses[k].position.contiguous()
                        current_quat = link_poses[k].quaternion.contiguous()

                        pose_err, pos_err, quat_err = current_fn.forward_out_distance(
                            current_pos, current_quat, self._goal_buffer, k
                        )
                        pose_error.append(pose_err)
                        position_error.append(pos_err)
                        quat_error.append(quat_err)
            out_metrics.pose_error = cat_max(pose_error)
            out_metrics.rotation_error = cat_max(quat_error)
            out_metrics.position_error = cat_max(position_error)

        if (
            self._goal_buffer.goal_state is not None
            and self.convergence_cfg.cspace_cfg is not None
            and self.cspace_convergence.enabled
        ):
            _, out_metrics.cspace_error = self.cspace_convergence.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state.state_seq.position,
                self._goal_buffer.batch_goal_state_idx,
                True,
            )

        if (
            self.convergence_cfg.null_space_cfg is not None
            and self.null_convergence.enabled
            and self._goal_buffer.batch_retract_state_idx is not None
        ):
            out_metrics.null_space_error = self.null_convergence.forward_target_idx(
                self._goal_buffer.retract_state,
                state.state_seq.position,
                self._goal_buffer.batch_retract_state_idx,
            )

        if self.admittance_cost is not None and self.admittance_cost.enabled:
            adm_err = self.admittance_cost.last_error()
            if adm_err is not None:
                out_metrics.admittance_error = adm_err  # NEW added: 임피던스 오차를 메트릭으로 노출. (한국어 주석)

        return out_metrics

    def update_params(
        self,
        goal: Goal,
    ):
        """
        Update params for the cost terms and dynamics model.

        """

        super(ArmReacher, self).update_params(goal)
        if goal.batch_pose_idx is not None:
            self._goal_idx_update = False
        if goal.goal_pose.position is not None:
            self.enable_cspace_cost(False)
        return True

    def update_contact_measurement(
        self,
        contact_wrench: Optional[torch.Tensor],
        position_error: Optional[torch.Tensor] = None,
        velocity_error: Optional[torch.Tensor] = None,
        acceleration_error: Optional[torch.Tensor] = None,
    ) -> None:
        """Cache the latest measured wrench and optional Cartesian errors."""

        if contact_wrench is None:
            # NEW added: 접촉이 없으면 캐시를 초기화해 불필요한 비용 계산을 방지. (한국어 주석)
            self._contact_wrench = None
            self._contact_cartesian_error = None
            return

        wrench = self.tensor_args.to_device(contact_wrench)
        if wrench.dim() == 1:
            wrench = wrench.view(1, -1)
        self._contact_wrench = wrench  # NEW added: 최신 렌치를 GPU 텐서로 저장. (한국어 주석)

        cartesian_chunks = []  # NEW added: 위치/속도/가속 오차를 순서대로 누적. (한국어 주석)
        for item in (position_error, velocity_error, acceleration_error):
            if item is None:
                cartesian_chunks.append(
                    torch.zeros_like(wrench[:, :6])
                )  # NEW added: 결측값은 영벡터로 채워 임피던스 입력을 안정화. (한국어 주석)
            else:
                tensor_item = self.tensor_args.to_device(item)
                if tensor_item.dim() == 1:
                    tensor_item = tensor_item.view(1, -1)
                cartesian_chunks.append(tensor_item)  # NEW added: 사용자 제공 오차를 그대로 사용. (한국어 주석)

        self._contact_cartesian_error = torch.cat(cartesian_chunks, dim=-1)
        # NEW added: 위치/속도/가속 오차를 하나의 벡터로 결합해 비용 함수에 전달. (한국어 주석)
        
    def enable_pose_cost(self, enable: bool = True):
        if enable:
            self.goal_cost.enable_cost()
        else:
            self.goal_cost.disable_cost()

    def enable_cspace_cost(self, enable: bool = True):
        if enable:
            self.dist_cost.enable_cost()
            self.cspace_convergence.enable_cost()
        else:
            self.dist_cost.disable_cost()
            self.cspace_convergence.disable_cost()

    def get_pose_costs(
        self,
        include_link_pose: bool = False,
        include_convergence: bool = True,
        only_convergence: bool = False,
    ):
        if only_convergence:
            return [self.pose_convergence]
        pose_costs = [self.goal_cost]
        if include_convergence:
            pose_costs += [self.pose_convergence]
        if include_link_pose:
            log_error("Not implemented yet")
        return pose_costs

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        pose_costs = self.get_pose_costs(
            include_link_pose=metric.include_link_pose, include_convergence=False
        )
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=True)

        pose_costs = self.get_pose_costs(only_convergence=True)
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=False)


@get_torch_jit_decorator()
def cat_sum_reacher(tensor_list: List[torch.Tensor]):
    valid_tensors: List[torch.Tensor] = []
# <<<<<<< aciui4-codex/update-yaml-file-loaders-for-utf-8-encoding
    reference_tensor: Optional[torch.Tensor] = None
    fallback_tensor: Optional[torch.Tensor] = None

    for tensor in tensor_list:
        if fallback_tensor is None:
            fallback_tensor = tensor

        if tensor.numel() == 0:
            continue

        if reference_tensor is None:
            reference_tensor = tensor

        if (
            reference_tensor is not None
            and tensor.shape == reference_tensor.shape
            and tensor.dtype == reference_tensor.dtype
            and tensor.device == reference_tensor.device
        ):
            valid_tensors.append(tensor)

    if reference_tensor is None:
        if fallback_tensor is None:
            return torch.tensor(0.0)
        return torch.zeros_like(fallback_tensor)
# =======
#     if len(tensor_list) > 0:
#         reference_tensor: torch.Tensor = tensor_list[0]
#     else:
#         reference_tensor = torch.zeros(0)

#     for tensor in tensor_list:
#         if tensor.numel() > 0:
#             valid_tensors.append(tensor)

#     if len(valid_tensors) == 0:
#         return torch.zeros_like(reference_tensor)
# >>>>>>> main

    cat_tensor = torch.sum(torch.stack(valid_tensors, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon_reacher(tensor_list: List[torch.Tensor]):
    valid_tensors: List[torch.Tensor] = []
# <<<<<<< aciui4-codex/update-yaml-file-loaders-for-utf-8-encoding
    reference_tensor: Optional[torch.Tensor] = None
    fallback_tensor: Optional[torch.Tensor] = None

    for tensor in tensor_list:
        if fallback_tensor is None:
            fallback_tensor = tensor

        if tensor.numel() == 0 or tensor.dim() == 0:
            continue

        if reference_tensor is None:
            reference_tensor = tensor

        if (
            reference_tensor is not None
            and tensor.shape == reference_tensor.shape
            and tensor.dtype == reference_tensor.dtype
            and tensor.device == reference_tensor.device
        ):
            valid_tensors.append(tensor)

    if reference_tensor is None:
        if fallback_tensor is None:
            return torch.tensor(0.0)
        if fallback_tensor.dim() == 0:
            return torch.zeros_like(fallback_tensor)
        zero_tensor = torch.zeros_like(fallback_tensor)
        return torch.sum(zero_tensor, dim=-1)
# =======
#     if len(tensor_list) > 0:
#         reference_tensor: torch.Tensor = tensor_list[0]
#     else:
#         reference_tensor = torch.zeros(0)

#     for tensor in tensor_list:
#         if tensor.numel() > 0:
#             valid_tensors.append(tensor)

#     if len(valid_tensors) == 0:
#         return torch.zeros_like(reference_tensor)
# >>>>>>> main

    cat_tensor = torch.sum(torch.stack(valid_tensors, dim=0), dim=(0, -1))
    return cat_tensor
