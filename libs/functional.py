from typing import List, Union

import numpy as np


class Variable:

    def __init__(self, value=None, grad=None):
        self.value = value
        self.grad = grad


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        value = 0
        grad = 0
        reverse = np.flipud(self.a)  # 把陣列反轉
        poly = np.poly1d(reverse)  # 把陣列轉為多項式函數
        deriv = poly.deriv()  # 把多項式函數微分
        value = poly(x)  # 多項式函數求值
        grad = deriv(x)  # 微分後求值
        result = Variable(value, grad)
        return result
