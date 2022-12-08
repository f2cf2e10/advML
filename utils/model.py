from abc import ABC, abstractmethod

import numpy as np

from utils.types import Model, Norm


class LinearRegression(Model):

    def __init__(self, params: np.ndarray, with_const: bool):
        self.theta = None
        self.set_theta(params)
        self.with_const = with_const

    def set_theta(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_x(self, x: np.ndarray) -> np.ndarray:
        return x

    def value(self, x: np.ndarray) -> float:
        return self.theta.dot(self.get_x(x))

    def adversarial_value(self, x: np.ndarray, xi: float, norm: Norm) -> np.ndarray:
        theta = self.get_theta()
        dual_norm = norm.dual(theta)
        return np.array([self.theta.dot(self.get_x(x)) + xi * dual_norm])

    def dtheta(self, x: np.ndarray) -> np.ndarray:
        return self.get_x(x)

    def dx(self, x: np.ndarray) -> np.ndarray:
        return self.theta

    def copy(self):
        other = LinearRegression(self.get_theta().copy(), self.with_const)
        return other


class LinearClassifier(Model):
    def __init__(self, params: np.ndarray, with_const: bool):
        self.theta = None
        self.set_theta(params)
        self.with_const = with_const

    def set_theta(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_x(self, x: np.ndarray) -> np.ndarray:
        return x

    def value(self, x: np.ndarray) -> float:
        return 1 * (self.theta.dot(self.get_x(x)) > 0)

    def adversarial_value(self, x: np.ndarray, xi: float, norm: Norm) -> float:
        raise NotImplementedError("Linear classifier derivative not implemented")

    def dtheta(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Linear classifier derivative not implemented")

    def dx(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Linear classifier derivative not implemented")

    def copy(self):
        other = LinearClassifier(self.get_theta().copy(), self.with_const)
        return other


class LogisticRegression(Model):
    def __init__(self, params: np.ndarray, with_const: bool):
        self.theta = None
        self.set_theta(params)
        self.with_const = with_const

    def set_theta(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_x(self, x: np.ndarray) -> np.ndarray:
        return x

    def value(self, x: np.ndarray) -> float:
        return 1. / (1. + np.exp(-self.theta.dot(self.get_x(x))))

    def adversarial_value(self, x: np.ndarray, xi: float, norm: Norm) -> np.ndarray:
        dual_norm = norm.dual(self.get_theta())
        return np.array([1. / (1. + np.exp(-self.theta.dot(self.get_x(x) + dual_norm)))])

    def dtheta(self, x: np.ndarray) -> np.ndarray:
        return self.get_x(x) * np.exp(-self.theta.dot(self.get_x(x))) / (
                1. + np.exp(- self.theta.dot(self.get_x(x)))) ** 2

    def dx(self, x: np.ndarray) -> np.ndarray:
        return self.theta * np.exp(-self.theta.dot(self.get_x(x))) / (
                1. + np.exp(- self.theta.dot(self.get_x(x)))) ** 2

    def copy(self):
        other = LogisticRegression(self.get_theta().copy(), self.with_const)
        return other


def SVM(Model):
    def __init__(self, params: np.ndarray, with_const: bool):
        self.theta = None
        self.set_theta(params)
        self.with_const = with_const

    def set_theta(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_x(self, x: np.ndarray) -> np.ndarray:
        return x

    def value(self, x: np.ndarray) -> float:
        return 1. / (1. + np.exp(-self.theta.dot(self.get_x(x))))

    def adversarial_value(self, x: np.ndarray, xi: float, norm: Norm) -> np.ndarray:
        dual_norm = norm.dual(self.get_theta())
        return np.array([1. / (1. + np.exp(-self.theta.dot(self.get_x(x) + dual_norm)))])

    def dtheta(self, x: np.ndarray) -> np.ndarray:
        return self.get_x(x) * np.exp(-self.theta.dot(self.get_x(x))) / (
                1. + np.exp(- self.theta.dot(self.get_x(x)))) ** 2

    def dx(self, x: np.ndarray) -> np.ndarray:
        return self.theta * np.exp(-self.theta.dot(self.get_x(x))) / (
                1. + np.exp(- self.theta.dot(self.get_x(x)))) ** 2

    def copy(self):
        other = LogisticRegression(self.get_theta().copy(), self.with_const)
        return other
