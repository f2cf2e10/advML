import numpy as np
from cvxopt import matrix, solvers

from utils.types import Norm


class L1(Norm):
    @staticmethod
    def norm(x: np.ndarray) -> float:
        return sum(np.abs(x))

    @staticmethod
    def norm_dx(x: np.ndarray) -> float:
        return np.sign(x)

    @staticmethod
    def dual(x: np.ndarray) -> float:
        return Linf.norm(x)

    @staticmethod
    def dual_dx(x: np.ndarray) -> float:
        return Linf.norm_dx(x)

    @staticmethod
    def proj(x0: np.ndarray, constraint: float) -> np.ndarray:
        N = len(x0)
        P = matrix([[matrix(np.eye(N)), matrix(np.zeros(N * N).reshape([N, N]))],
                    [matrix(np.zeros(N * N).reshape([N, N])), matrix(np.zeros(N * N).reshape([N, N]))]])
        q = matrix([matrix(-2 * x0), matrix(np.zeros(N))])
        G = matrix([[matrix(np.eye(N)), -matrix(np.eye(N))],
                    [-matrix(np.eye(N)), -matrix(np.eye(N))],
                    [matrix(np.zeros(N * N).reshape([N, N])), matrix(-np.eye(N))],
                    [matrix(np.zeros(N)), matrix(np.ones(N))]]).T
        h = matrix([0.0] * 3 * N + [constraint])
        sol = solvers.qp(P, q, G, h)
        return np.array(sol['x'][0:N].T)[0]


class L2(Norm):
    @staticmethod
    def norm(x: np.ndarray) -> float:
        return np.sum(x ** 2) ** 0.5

    @staticmethod
    def norm_dx(x: np.ndarray) -> float:
        return x / L2.norm(x)

    @staticmethod
    def dual(x: np.ndarray) -> float:
        return L2.norm(x)

    @staticmethod
    def dual_dx(x: np.ndarray) -> float:
        return L2.norm_dx(x)

    @staticmethod
    def proj(x0: np.ndarray, constraint: float) -> np.ndarray:
        return x0 / L2.norm(x0) * constraint


class Linf(Norm):
    @staticmethod
    def norm(x: np.ndarray) -> float:
        return np.max(np.abs(x))

    @staticmethod
    def norm_dx(x: np.ndarray) -> float:
        raise NotImplementedError("Not Implemented")

    @staticmethod
    def dual(x: np.ndarray) -> float:
        return L1.norm(x)

    @staticmethod
    def dual_dx(x: np.ndarray) -> float:
        return L1.norm_dx(x)

    @staticmethod
    def proj(x0: np.ndarray, constraint: float) -> np.ndarray:
        return np.clip(x0, -constraint, constraint)
