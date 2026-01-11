from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)

def axis_angle_to_rot(axis_angle: np.ndarray) -> np.ndarray:
    w = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = _skew(k)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    tr = max(-1.0, min(3.0, tr))
    theta = math.acos(max(-1.0, min(1.0, 0.5 * (tr - 1.0))))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]], dtype=np.float64)
    return 0.5 * theta / math.sin(theta) * w

@dataclass(frozen=True)
class SE3:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> "SE3":
        Rt = self.R.T
        tt = -(Rt @ self.t)
        return SE3(R=Rt, t=tt)

    def __matmul__(self, other: "SE3") -> "SE3":
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return SE3(R=R, t=t)

def project_points(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3,3)
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1,3)
    z = np.clip(pts[:,2:3], 1e-6, None)
    xy = pts[:, :2] / z
    uv = (K[:2,:2] @ xy.T).T + K[:2, 2]
    return uv
