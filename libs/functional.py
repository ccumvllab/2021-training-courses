from typing import List, Union, Tuple

import numpy as np


class Variable:

    def __init__(self, value=None):
        self.value = value
        self.grad = None


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        pass


def regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:
    pass
