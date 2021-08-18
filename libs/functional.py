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
    m = np.zeros(num_iterations+1)
    b = np.zeros(num_iterations+1)
    m[0] = np.random.rand()
    b[0] = np.random.rand()
    for i in range(num_iterations):
        #隨機抽取batch_size個sample
        sample_label = [j for j in range(num_samples)]
        sample_label = np.random.choice(sample_label,batch_size,replace=False)
        batch_x = x[sample_label]
        batch_y = y[sample_label]
        m[i+1] = m[i]
        b[i+1] = b[i]
        #GD
        for j in range(batch_size):
            m[i+1] -= learning_rate*batch_x[j]*(m[i+1]*batch_x[j]+-batch_y[j])/batch_size # 2x(mx+b-y)
            b[i+1] -= learning_rate*(m[i+1]*batch_x[j]+-batch_y[j])/batch_size #2(mx+b-y)
    return (m,b)

