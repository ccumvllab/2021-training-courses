from pathlib import Path

import numpy as np

from libs.functional import Polynomial
from libs.visualization import plot_polynomial


class TestPlotPolynomial:

    def test_plot_polynomial_1(self):
        f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
        file = Path('test_plot_polynomial_1.png')
        x = np.linspace(-4.3, 3.7)
        plot_polynomial(file=file, function=f, p_x=1., x=x, formula='1 + 2x + 3x^2')
        assert file.exists()

    def test_plot_polynomial_2(self):
        f = Polynomial(a=[4., 1., 3., 1.2])  # 4 + x + 3x^2 + 1.2x^3
        file = Path('test_plot_polynomial_2.png')
        x = np.linspace(-2.5, 1.0)
        plot_polynomial(file=file, function=f, p_x=-1., x=x, formula='4 + x + 3x^2 + 1.2x^3')
        assert file.exists()
