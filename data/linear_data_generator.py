import numpy as np
from typing import List
from utils.types import Data


def _initialize(seed: int, with_const: float):
    np.random.seed(seed)
    a = 2 * np.random.rand(1) - 1
    b = 2 * np.random.rand(1) - 1 if with_const else 0.0
    return a, b


def _generate_synthetic_linear_model_sample_with_uniform_distr(a: float, b: float, noise: float, m: int,
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


def _generate_synthetic_linear_model_sample_with_cap_normal_distr(a: float, b: float, noise: float, m: int,
                                                                  with_const: bool = False, max_val: float = 1.,
                                                                  std: float = 0.25) -> List[Data]:
    # data generation
    lm = lambda x: np.float64(1.0) if x[1] > a * x[0] + b + noise * np.random.randn() else np.float64(-1.)
    x = np.clip(std * np.random.randn(2 * m).reshape([m, 2]), -max_val, max_val)
    if with_const:
        x = np.hstack([x, np.ones(m).reshape(m, 1)])
    y = [lm(x_i) for x_i in x]
    data = [{'x': x[i], 'y': y[i]} for i in range(len(x))]
    d = len(data[0].get('x'))
    return data, d, x, y


def generate_synthetic_linear_model_with_uniform_distr_samples(noise: float, m: int, N: int, seed: int = 7777,
                                                               with_const: bool = False) -> List[List[Data]]:
    # data generation
    a, b = _initialize(seed, False)
    data = []
    for i in range(N):
        data_i, _, _, _ = _generate_synthetic_linear_model_sample_with_uniform_distr(a, b, noise, m, with_const)
        data += [data_i]
    return data


def generate_synthetic_linear_model_with_uniform_distr_sample(noise: float, m: int, seed: int = 7777,
                                                              with_const: bool = False) -> List[Data]:
    a, b = _initialize(seed, False)
    data, d, x, y = _generate_synthetic_linear_model_sample_with_uniform_distr(a, b, noise, m, with_const)
    return a, b, d, data, x, y


def generate_synthetic_linear_model_with_cap_normal_distr_samples(noise: float, m: int, N: int, seed: int = 7777,
                                                                  with_const: bool = False) -> List[List[Data]]:
    # data generation
    a, b = _initialize(seed, False)
    data = []
    for i in range(N):
        data_i, _, _, _ = _generate_synthetic_linear_model_sample_with_cap_normal_distr(a, b, noise, m, with_const)
        data += [data_i]
    return data


def generate_synthetic_linear_model_with_cap_normal_distr_sample(noise: float, m: int, seed: int = 7777,
                                                                 with_const: bool = False) -> List[Data]:
    a, b = _initialize(seed, False)
    data, d, x, y = _generate_synthetic_linear_model_sample_with_cap_normal_distr(a, b, noise, m, with_const)
    return a, b, d, data, x, y
