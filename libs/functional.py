from typing import List, Union

import numpy as np


class Variable:

    def __init__(self, value=None):
        self.value = value
        self.grad = None


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        f = np.power(x, range(len(self.a)))
        var = Variable(value=sum(self.a * f))

        coef = np.arange(1, len(self.a))
        grad_f = np.power(x, range(len(self.a) - 1))
        var.grad = sum(coef * self.a[1:] * grad_f)
        return var
