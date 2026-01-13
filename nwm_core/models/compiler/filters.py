from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class Gaussian:
    mean: torch.Tensor  # (D,)
    cov: torch.Tensor   # (D,D)


class CVFilter:
    def __init__(
        self,
        init_mean: torch.Tensor,     # (6,)
        init_cov: torch.Tensor,      # (6,6)
        q_pos: float = 2e-3,
        q_vel: float = 5e-3,
        device: torch.device | None = None,
    ):
        self.device = device or init_mean.device
        self.state = Gaussian(init_mean.to(self.device), init_cov.to(self.device))
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)

    def predict(self, dt: float) -> None:
        dt = float(max(dt, 1e-4))
        F = torch.eye(6, device=self.device)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = torch.zeros((6, 6), device=self.device)
        Q[0, 0] = self.q_pos
        Q[1, 1] = self.q_pos
        Q[2, 2] = self.q_pos
        Q[3, 3] = self.q_vel
        Q[4, 4] = self.q_vel
        Q[5, 5] = self.q_vel

        m = F @ self.state.mean
        P = F @ self.state.cov @ F.T + Q
        self.state = Gaussian(m, P)

    def update(self, z: torch.Tensor, R: torch.Tensor) -> None:
        # z: (3,) measurement for [cx,cy,z]
        z = z.to(self.device)
        R = R.to(self.device)

        H = torch.zeros((3, 6), device=self.device)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        m = self.state.mean
        P = self.state.cov

        y = z - (H @ m)
        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)

        m_new = m + K @ y
        P_new = (torch.eye(6, device=self.device) - K @ H) @ P
        self.state = Gaussian(m_new, P_new)
