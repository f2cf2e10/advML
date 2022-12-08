import random
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, TypeVar
from typing_extensions import TypedDict

import numpy as np

Data = TypedDict('Data', {'x': np.ndarray,
                          'y': np.float64})

ModelType = TypeVar('ModelType', bound='Model')


class Norm(ABC):
    @staticmethod
    @abstractmethod
    def norm(x: np.ndarray) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def norm_dx(x: np.ndarray) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def dual(x: np.ndarray) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def dual_dx(x: np.ndarray) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def proj(x: np.ndarray, x0: np.ndarray, constraint: float) -> np.ndarray:
        raise NotImplementedError("Abstract class method")


class Model(ABC):
    @abstractmethod
    def value(self, x: np.ndarray) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def adversarial_value(self, x: np.ndarray, xi: np.float64, norm: Norm) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def get_x(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def dtheta(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def dx(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def copy(self) -> ModelType:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def get_theta(self) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def set_theta(self, theta: np.ndarray):
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def get_x(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract class method")


class Loss(ABC):
    @abstractmethod
    def f(self, model: Model, data: Data) -> np.float64:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def dtheta(self, model: Model, data: Data) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def dx(self, model: Model, data: Data) -> np.ndarray:
        raise NotImplementedError("Abstract class method")

    @abstractmethod
    def dy(self, model: Model, data: Data) -> float:
        raise NotImplementedError("Abstract class method")

    def f_batch(self, model: Model, data: List[Data]) -> np.float64:
        return np.float64(reduce(lambda x, y: x + y, [self.f(model, data_i) for data_i in data]) / len(data))

    def f_mini_batch(self, model: Model, data: List[Data], batch_size: int) -> np.float64:
        return np.float64(reduce(lambda x, y: x + y,
                      [self.f(model, data_i) for data_i in random.sample(data, batch_size)]) / batch_size)

    def f_stochastic(self, model: Model, data: List[Data]) -> np.float64:
        return self.f(model, random.sample(data, 1)[0])

    def dtheta_batch(self, model: Model, data: List[Data]) -> np.ndarray:
        return reduce(lambda x, y: x + y, [self.dtheta(model, data_i) for data_i in data]) / len(data)

    def dtheta_mini_batch(self, model: Model, data: List[Data], batch_size: int) -> np.ndarray:
        return reduce(lambda x, y: x + y,
                      [self.dtheta(model, data_i) for data_i in random.sample(data, batch_size)]) / batch_size

    def dtheta_stochastic(self, model: Model, data: List[Data]) -> np.ndarray:
        return self.dtheta(model, random.sample(data, 1)[0])


class Attack(ABC):
    @staticmethod
    @abstractmethod
    def adversarial_example(loss: Loss, model: Model, data: Data, xi: float, *args, **kwargs) -> Data:
        raise NotImplementedError("Abstract class method")
