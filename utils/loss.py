from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .model import Model
from .types import Data, Loss


class L2(Loss):

    def f(self, model: Model, data: Data) -> np.float64:
        y = data.get("y")
        x = data.get("x")
        return np.float64(0.5 * (model.value(x) - y) ** 2)

    def dtheta(self, model: Model, data: Data) -> np.ndarray:
        y = data.get("y")
        x = data.get("x")
        return (model.value(x) - y) * model.dtheta(x)

    def dx(self, model: Model, data: Data) -> np.ndarray:
        y = data.get("y")
        x = data.get("x")
        return (model.value(x) - y) * model.dx(x)

    def dy(self, model: Model, data: Data) -> float:
        y = data.get("y")
        x = data.get("x")
        return -(model.value(x) - y)


class CrossEntropy(Loss):

    def f(self, model: Model, data: Data) -> np.float64:
        y = data.get("y")
        x = data.get("x")
        return np.float64(np.log(1 + np.exp(-y * model.value(x))))

    def dtheta(self, model: Model, data: Data) -> np.ndarray:
        y = data.get("y")
        x = data.get("x")
        return -y * model.dtheta(x) / (1 + np.exp(y * model.value(x)))

    def dx(self, model: Model, data: Data) -> np.ndarray:
        y = data.get("y")
        x = data.get("x")
        return -y * model.dx(x) / (1 + np.exp(y * model.value(x)))

    def dy(self, model: Model, data: Data) -> float:
        y = data.get("y")
        x = data.get("x")
        return -model.value(x) / (1 + np.exp(y * model.value(x)))
