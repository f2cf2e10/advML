import numpy as np

from adversary.solver import _build_constraint_matrix, robust_adv_data_driven_binary_classifier


def test_constraint_matrix_1_1():
    x = np.array([1.0])
    y = 0.5
    xi = 2.0
    data = [{'x': x, 'y': y}]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_1_2():
    x = np.array([1.0, 2.0])
    y = 0.5
    xi = 2.0
    data = [{'x': x, 'y': y}]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_1_rand():
    x = np.random.rand(np.random.randint(5, 20))
    y = 0.5
    xi = 2.0
    data = [{'x': x, 'y': y}]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_2_1():
    x = np.array([[1.0], [2.0]])
    y = np.array([[0.5], [1.0]])
    xi = 2.0
    data = [{'x': x[i, :], 'y': y[i, 0]} for i in range(x.shape[0])]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_2_2():
    x = np.array([[1.0, 2.0], [2.0, 3.0]])
    y = np.array([[0.5], [1.0]])
    xi = 2.0
    data = [{'x': x[i, :], 'y': y[i, 0]} for i in range(x.shape[0])]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_2_rand():
    N = np.random.randint(5, 20)
    x = np.random.rand(2 * N).reshape(2, N)
    y = np.array([[0.5], [1.0]])
    xi = 2.0
    data = [{'x': x[i, :], 'y': y[i, 0]} for i in range(x.shape[0])]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8


def test_constraint_matrix_rand_rand():
    N = np.random.randint(5, 20)
    M = np.random.randint(5, 20)
    x = np.random.rand(M * N).reshape(M, N)
    y = np.random.rand(M).reshape(M, 1)
    xi = 2.0
    data = [{'x': x[i, :], 'y': y[i, 0]} for i in range(x.shape[0])]
    matrix = _build_constraint_matrix(xi, data)
    assert np.max(np.abs(matrix - (-y / xi * x))) <= 1E-8

def test_solver():
    N = np.random.randint(5, 20)
    M = np.random.randint(5, 20)
    x = np.random.rand(M * N).reshape(M, N)
    y = np.random.rand(M).reshape(M, 1)
    xi = 2.0
    data = [{'x': x[i, :], 'y': y[i, 0]} for i in range(x.shape[0])]
    w, _ = robust_adv_data_driven_binary_classifier(xi, data)
    print(w)
    assert np.abs(np.sum(np.abs(w)) - 1.0) < 1E-7
