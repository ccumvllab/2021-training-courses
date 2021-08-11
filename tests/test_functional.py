from libs.functional import Polynomial


class TestPolynomial:

    def test_io(self):
        f = Polynomial(a=[1, 2, 3])  # 1 + 2x + 3x^2
        assert f(4).value == 57
        assert f(5).value == 86

    def test_grad(self):
        f = Polynomial(a=[1, 2, 3])  # 1 + 2x + 3x^2
        # gradient: <2 + 6x>
        assert f(4).grad == 26
        assert f(5).grad == 32
        
    def test_plot_polynomial(self):
        f = Polynomial(a=[1, 2, 3])
        assert f(4).grad == 26
        assert f(4).value == 57
        assert f(5).value == 86
        assert f(5).grad == 32
        plot_polynomial(7,a=[1, 2, 3])
