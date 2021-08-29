from typing import List, Union, Tuple

import numpy as np
import random
from tqdm import tqdm


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

    # declare m and b
    m = np.empty(num_iterations + 1, dtype=np.float64)
    b = np.empty(num_iterations + 1, dtype=np.float64)

    # init m and b
    m[0] = random.random()
    b[0] = random.random()

    # split into batches
    x = np.array_split(x, batch_size)
    y = np.array_split(y, batch_size)

    for iteration in tqdm(range(num_iterations)):
        for batch, data in enumerate(x):

            # get gradient of m and b
            m_gradient = 1/num_samples * np.sum(
                2*data*(m[iteration]*data+b[iteration]-y[batch])
            )
            b_gradient = 1/num_samples * np.sum(
                2*(m[iteration]*data+b[iteration]-y[batch])
            )

            # update parameters by SGD
            m[iteration+1] = m[iteration] - learning_rate * m_gradient
            b[iteration+1] = b[iteration] - learning_rate * b_gradient

    print(f"the final parameters: (m, b)=({m[-1]}, {b[-1]})")

    return m, b
