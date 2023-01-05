import numpy as np
from typing import List
from utils.types import Data


def _generate_synthetic_linear_model_sample_with_uniform_distr(a: float, b: float, noise: float, m: int) -> List[Data]:
    # data generation
    np.random.seed(1771)
    lm = lambda x: np.float64(1.0) if x[1] > a * x[0] + b + noise * np.random.randn() else np.float64(-1.)
    x = (2 * np.random.rand(2 * m) - 1).reshape([m, 2])
    y = [lm(x_i) for x_i in x]
    data = [{'x': x[i], 'y': y[i]} for i in range(len(x))]
    d = len(data[0].get('x'))
    return data, d, x, y


def _generate_synthetic_linear_model_sample_with_cap_normal_distr(a: float, b: float, noise: float, m: int,
                                                                  max_val: float = 1., std: float = 0.25) -> List[Data]:
    # data generation
    np.random.seed(1771)
    lm = lambda x: np.float64(1.0) if x[1] > a * x[0] + b + noise * np.random.randn() else np.float64(-1.)
    x = np.clip(std * np.random.randn(2 * m).reshape([m, 2]), -max_val, max_val)
    y = [lm(x_i) for x_i in x]
    data = [{'x': x[i], 'y': y[i]} for i in range(len(x))]
    d = len(data[0].get('x'))
    return data, d, x, y


def generate_synthetic_linear_model_with_uniform_distr_samples(noise: float, m: int, N: int, a: float = None,
                                                               b: float = None) -> List[List[Data]]:
    # data generation
    data, _, _, _ = _generate_synthetic_linear_model_sample_with_uniform_distr(a, b, noise, m * N)
    return [data[i * m:(i + 1) * m] for i in range(N)]


def generate_synthetic_linear_model_with_uniform_distr_sample(noise: float, m: int, a: float, b: float) -> \
        List[List[Data]]:
    # data generation
    data, d, x, y = _generate_synthetic_linear_model_sample_with_uniform_distr(a, b, noise, m)
    return a, b, d, data, x, y


def generate_synthetic_linear_model_with_cap_normal_distr_samples(noise: float, m: int, N: int, a: float, b: float,
                                                                  max_val: float = 1., std: float = 0.25) -> \
        List[List[Data]]:
    # data generation
    data, _, _, _ = _generate_synthetic_linear_model_sample_with_cap_normal_distr(a, b, noise, m * N, max_val, std)
    return [data[i * m:(i + 1) * m] for i in range(N)]


def generate_synthetic_linear_model_with_cap_normal_distr_sample(noise: float, m: int, a: float, b: float,
                                                                 max_val: float = 1., std: float = 0.25) -> \
        List[List[Data]]:
    # data generation
    data, d, x, y = _generate_synthetic_linear_model_sample_with_cap_normal_distr(a, b, noise, m, max_val, std)
    return a, b, d, data, x, y
