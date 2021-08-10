from pathlib import Path

from libs.functional import Polynomial
from libs.visualization import plot_polynomial


def test_plot_polynomial():
    f = Polynomial(a=[1., 2., 3.])  # 1 + 2x + 3x^2
    file = Path('test_plot_polynomial.png')
    plot_polynomial(file=file, function=f, x=4.)
    assert file.exists()
