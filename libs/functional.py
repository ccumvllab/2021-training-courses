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
        if isinstance(x, np.ndarray):
            _x = x.repeat(len(self.a)).reshape(len(x), len(self.a))
            f = np.power(_x, range(len(self.a)))
            var = Variable(value=np.sum(self.a * f, 1))

            coef = np.arange(1, len(self.a))
            grad_f = np.power(_x[:, :-1], range(len(self.a) - 1))
            var.grad = np.sum(coef * self.a[1:] * grad_f, 1)
            return var
        else:
            f = np.power(x, range(len(self.a)))
            var = Variable(value=sum(self.a * f))

            coef = np.arange(1, len(self.a))
            grad_f = np.power(x, range(len(self.a) - 1))
            var.grad = sum(coef * self.a[1:] * grad_f)
            return var
