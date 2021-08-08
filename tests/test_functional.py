from typing import List, Union
import math
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size


class Variable:

    def __init__(self, value=None, grad=None):
        self.value = value
        self.grad = grad


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x) -> Variable:
        result_value = 0
        result_grad = 0
        for i in range(len(self.a)):
            # 多項式計算
            result_value += self.a[i]*math.pow(x, i)
            # 微分計算
            if(i > 0):  # 第一項為常數因此直接忽略
                result_grad += self.a[i]*math.pow(x*i, i-1)

        x_var = Variable(result_value, result_grad)  # 計算結果存入 Variable Class
        return x_var


class TestPolynomial:
    def __init__(self, a: List):
        self.function = Polynomial(a)  # 定義測試多項式參數

    def test_io(self):  # 多項式計算測試
        assert self.function(4).value == 57
        assert self.function(5).value == 86

    def test_grad(self):  # 微分計算測試
        assert self.function(4).grad == 26
        assert self.function(5).grad == 32

    def line(self, x, x1, y1):  # 計算切線
        return self.function(x1).grad*(x - x1) + y1

    def plot_polynomial(self,  a: List = None):
        xpt = np.linspace(0, 9, 10)  # x 軸數值
        ypt = []  # y軸數值
        for i in xpt:
            ypt.append(self.function(i).value)

        plt.plot(xpt, self.line(xpt, 5, self.function(5).value),
                 linewidth=1)  # 畫線
        plt.plot(xpt, ypt)  # 畫線
        plt.savefig('./tests/TestPolynomial.png')
        plt.ylim(0, 250)
        plt.xlim(0, 9)
        plt.show()  # 顯示繪製的圖形


if __name__ == '__main__':
    Test = TestPolynomial(a=[1, 2, 3])  # 1 + 2x + 3x^2
    Test.test_io()  # 1 + 2x + 3x^2
    Test.test_grad()   # 1 + 2x + 3x^2 -> gradient: <2 + 6x>
    Test.plot_polynomial()
