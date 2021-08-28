from typing import List, Union, Tuple

import numpy as np


class Variable:

    def __init__(self, value=None, grad=None):
        self.value = value
        self.grad = grad


class Polynomial:

    def __init__(self, a: List = None):
        self.coef = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        x = np.array(x)
        base = np.repeat(x, len(self.coef)).reshape(x.size, len(self.coef))

        exponent = np.arange(0, len(self.coef))

        # get the power from base and exponent, power=base^exponent
        power = np.power(base, exponent)

        # get the result of the polynomial
        y = np.sum(np.multiply(self.coef, power), axis=1)

        # get the coefficient of the differential polynomial
        coef_diff = np.multiply(self.coef[1:], np.arange(1, len(self.coef)))

        base_diff = np.repeat(x, len(coef_diff)).reshape(
            x.size, len(coef_diff))
        exponent_diff = np.arange(0, len(coef_diff))

        # get power
        power_diff = np.power(base_diff, exponent_diff)

        # get the result of the polynomial
        y_diff = np.sum(np.multiply(coef_diff, power_diff), axis=1)

        return Variable(y, y_diff)


def regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:
    pass
