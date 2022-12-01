import numpy as np
from typing import List
from utils.types import Data


def _initialize(seed: int, with_const: float):
    np.random.seed(seed)
    a = np.random.rand(1)
    b = np.random.rand(1) if with_const else 0.0
    return a, b


def _generate_synthetic_linear_model_sample(a: float, b: float, noise: float, m: int,
                                            with_const: bool = False) -> List[Data]:
    # data generation
    lm = lambda x: np.float64(1.0) if x[1] > a * x[0] + b + noise * np.random.randn() else np.float64(-1.)
    x = (2 * np.random.rand(2 * m) - 1).reshape([m, 2])
    if with_const:
        x = np.hstack([x, np.ones(m).reshape(m, 1)])
    y = [lm(x_i) for x_i in x]
    data = [{'x': x[i], 'y': y[i]} for i in range(len(x))]
    d = len(data[0].get('x'))
    return data, d, x, y


def generate_synthetic_linear_model_samples(noise: float, m: int, N: int, seed: int = 171,
                                            with_const: bool = False) -> List[List[Data]]:
    # data generation
    a, b = _initialize(seed, with_const)
    data = []
    for i in range(N):
        data_i, _, _, _ = _generate_synthetic_linear_model_sample(a, b, noise, m, with_const)
        data += [data_i]
    return data


def generate_synthetic_linear_model(noise: float, m: int, seed: int = 171, with_const: bool = False) -> List[Data]:
    a, b = _initialize(seed, with_const)
    data, d, x, y = _generate_synthetic_linear_model_sample(a, b, noise, m, with_const)
    return a, b, d, data, x, y
