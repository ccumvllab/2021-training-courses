import numpy as np

from libs.functional import Polynomial, regression_sgd


class TestPolynomial:

    def test_io(self):
        f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
        assert f(4.).value == 57.
        assert f(5.).value == 86.

    def test_io_array(self):
        f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
        x = np.array([4., 5.])
        y = np.array([57., 86.])
        assert np.all(f(x).value == y)

    def test_grad(self):
        f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
        # gradient: <2 + 6x>
        assert f(4.).grad == 26.
        assert f(5.).grad == 32.


def test_regression_solved_correctly():
    m_gt, b_gt = 2.4, 1.0
    num_samples = 100
    x = np.random.uniform(-10, 10, num_samples)
    noise = np.random.normal(0, 1, num_samples)
    y = m_gt * x + b_gt + noise
    num_iterations = 1000
    batch_size = 10
    learning_rate = 0.001

    m, b = regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate)

    assert m.shape == (num_iterations + 1,) and b.shape == (num_iterations + 1,)
    assert np.isclose(m[-1], m_gt, 1e-01)
