from typing import List, Union

import numpy as np


class Variable:

    def __init__(self, value=None):
        self.value = value
        self.grad = None


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, np.ndarray]) -> Variable:
        if isinstance(x, float):
            x = np.array([x])

        x = x.repeat(len(self.a)).reshape(len(x), len(self.a))

        f = np.power(x, range(len(self.a)))
        coef = np.arange(1, len(self.a))
        grad_f = np.power(x[:, :-1], range(len(self.a) - 1))

        var = Variable(value=np.sum(self.a * f, 1))
        var.grad = np.sum(coef * self.a[1:] * grad_f, 1)
        return var
