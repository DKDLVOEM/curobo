# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Admittance style contact force tracking costs."""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from .cost_base import CostBase, CostConfig

# NEW added: 접촉 임피던스 추적 비용을 정의하는 전용 모듈 구현부. (한국어 주석)


# NEW added: 스칼라/리스트 입력을 통일된 텐서 형태로 변환하는 헬퍼 함수. (한국어 주석)
def _to_tensor(
    tensor_args,
    value: Optional[Union[float, Sequence[float], torch.Tensor]],
    length: int,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = tensor_args.to_device(value)
    if tensor.numel() == 1:
        tensor = tensor.expand(length)
    return tensor


# NEW added: 임피던스 비용 설정값을 구조적으로 관리하기 위한 데이터클래스. (한국어 주석)
@dataclass
class AdmittanceCostConfig(CostConfig):
    """Configuration for :class:`AdmittanceCost`.

    Attributes
    ----------
    desired_wrench: Optional[Union[float, Sequence[float], torch.Tensor]]
        Target wrench to realize in contact. Ordered as [Fx, Fy, Fz, Tx, Ty, Tz].
    stiffness: Optional[Union[float, Sequence[float], torch.Tensor]]
        Cartesian stiffness gain applied to position error contribution.
    damping: Optional[Union[float, Sequence[float], torch.Tensor]]
        Cartesian damping gain applied to velocity error contribution.
    mass: Optional[Union[float, Sequence[float], torch.Tensor]]
        Apparent mass (diagonal) used for acceleration contribution.
    activation_threshold: Optional[float]
        Minimum norm of measured wrench required before the cost becomes active.
    desired_velocity: Optional[Union[float, Sequence[float], torch.Tensor]]
        Reference twist that should be achieved while in contact (m/s, rad/s).
    desired_acceleration: Optional[Union[float, Sequence[float], torch.Tensor]]
        Reference twist rate that should be achieved while in contact.
    return_loss: bool
        When True, return the norm of the wrench tracking error alongside the weighted cost.
    """

    desired_wrench: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    stiffness: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    damping: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    mass: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    activation_threshold: Optional[float] = None
    desired_velocity: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    desired_acceleration: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    return_loss: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.desired_wrench = _to_tensor(self.tensor_args, self.desired_wrench, 6)
        self.stiffness = _to_tensor(self.tensor_args, self.stiffness, 6)
        self.damping = _to_tensor(self.tensor_args, self.damping, 6)
        self.mass = _to_tensor(self.tensor_args, self.mass, 6)
        self.desired_velocity = _to_tensor(self.tensor_args, self.desired_velocity, 6)
        self.desired_acceleration = _to_tensor(
            self.tensor_args, self.desired_acceleration, 6
        )
        if self.activation_threshold is not None:
            self.activation_threshold = self.tensor_args.to_device(self.activation_threshold)  # NEW added: 활성화 임계값을 텐서화. (한국어 주석)


# NEW added: 임피던스 추적 오차를 계산해 MPPI 비용으로 반환하는 구현체. (한국어 주석)
class AdmittanceCost(CostBase, AdmittanceCostConfig):
    """Admittance style cost penalising wrench tracking error."""

    def __init__(self, config: AdmittanceCostConfig):
        # NEW added: 설정을 복사하고 임피던스 추적에 필요한 내부 상태를 초기화한다. (한국어 주석)
        AdmittanceCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._zero = self.tensor_args.to_device(0.0)
        self._last_error_norm: Optional[torch.Tensor] = None  # NEW added: 마지막 오차 노름 캐시. (한국어 주석)

    def forward(
        self,
        measured_wrench: torch.Tensor,
        cartesian_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate admittance tracking cost.

        Parameters
        ----------
        measured_wrench: torch.Tensor
            Tensor containing measured wrench samples with shape ``[B, H, 6]`` or ``[B, 6]``.
        cartesian_state: Optional[torch.Tensor]
            Optional tensor capturing stacked ``[pos, vel, acc]`` error with shape ``[B, H, 18]``.
        """

        if measured_wrench is None:
            self._last_error_norm = None
            return self._zero

        if measured_wrench.dim() == 2:
            measured_wrench = measured_wrench.unsqueeze(1)

        desired = self.desired_wrench  # NEW added: 목표 렌치 텐서를 불러옴. (한국어 주석)
        if desired is not None:
            desired = desired.view(1, 1, -1)
            desired = desired.expand_as(measured_wrench)
        else:
            desired = torch.zeros_like(measured_wrench)

        # NEW added: 목표 렌치와의 차이를 기본 오차로 정의. (한국어 주석)
        error = measured_wrench - desired

        if cartesian_state is not None:
            if cartesian_state.dim() == 2:
                cartesian_state = cartesian_state.unsqueeze(1)
            pos_err, vel_err, acc_err = torch.split(cartesian_state, 6, dim=-1)
            if self.stiffness is not None:
                # NEW added: 위치 오차에 탄성 계수를 적용해 힘 추종을 강화. (한국어 주석)
                error = error + self.stiffness.view(1, 1, -1) * pos_err
            if self.damping is not None:
                # NEW added: 속도 오차에 감쇠 계수를 적용해 진동을 제어. (한국어 주석)
                error = error + self.damping.view(1, 1, -1) * vel_err
            if self.mass is not None:
                # NEW added: 가속도 오차에 등가 질량을 적용해 관성 효과를 모델링. (한국어 주석)
                error = error + self.mass.view(1, 1, -1) * acc_err
        if self.desired_velocity is not None:
            # NEW added: 목표 속도를 반영해 접촉 시 원하는 움직임을 유지. (한국어 주석)
            error = error - self.desired_velocity.view(1, 1, -1)
        if self.desired_acceleration is not None:
            # NEW added: 목표 가속도를 반영해 미세한 추종 성능을 확보. (한국어 주석)
            error = error - self.desired_acceleration.view(1, 1, -1)

        if self.activation_threshold is not None:
            wrench_norm = torch.linalg.norm(measured_wrench[..., :3], dim=-1, keepdim=True)
            mask = (wrench_norm >= self.activation_threshold).to(error.dtype)
            error = error * mask  # NEW added: 임계값 미만인 샘플은 비용에서 제외. (한국어 주석)

        # NEW added: 제곱합 비용으로 가중치를 적용해 샘플별 중요도를 계산. (한국어 주석)
        error_sq = error * error
        weight = self.weight.view(1, 1, -1)
        weighted = error_sq * weight
        cost = torch.sum(weighted, dim=-1)
        self._last_error_norm = torch.linalg.norm(error, dim=-1)  # NEW added: 이후 모니터링을 위한 오차 노름 저장. (한국어 주석)
        return cost

    def last_error(self) -> Optional[torch.Tensor]:
        """Return the most recent per-sample wrench error norm."""

        # NEW added: 최근 계산된 임피던스 오차 노름을 외부에서 조회할 수 있도록 제공. (한국어 주석)
        return self._last_error_norm
