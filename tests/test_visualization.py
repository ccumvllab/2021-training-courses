from pathlib import Path
import os,sys
sys.path.append('/Users/rayy1/Desktop/研究所/碩一/2021-training-courses')
import numpy as np

from libs.functional import Polynomial, regression_sgd
from libs.visualization import plot_polynomial, plot_regression


class TestPlotPolynomial:

    def test_plot_polynomial_1(self):
        f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
        file = Path('tests\\test_plot_polynomial_1.png')
        x = np.linspace(-4.3, 3.7)
        plot_polynomial(file=file, function=f, p_x=1., x=x, formula='1 + 2x + 3x^2')
        #assert file.exists()

    def test_plot_polynomial_2(self):
        f = Polynomial(a=[4., 1., 3., 1.2])  # 4 + x + 3x^2 + 1.2x^3
        file = Path('tests\\test_plot_polynomial_2.png')
        x = np.linspace(-2.5, 1.0)
        plot_polynomial(file=file, function=f, p_x=-1., x=x, formula='4 + x + 3x^2 + 1.2x^3')
        #assert file.exists()


def test_plot_regression():
    m_gt, b_gt = 2.4, 1.0
    num_samples = 100
    x = np.random.uniform(-10, 10, num_samples)
    noise = np.random.normal(0, 1, num_samples)
    y = m_gt * x + b_gt + noise
    num_iterations = 1000
    batch_size = 10
    learning_rate = 0.001

    m, b = regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate)
    file = Path('tests\\test_plot_regression.png')
    plot_regression(file, x, y, prediction=(m, b), groundtruth=(m_gt, b_gt))

    assert file.exists()
