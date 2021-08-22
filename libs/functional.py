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


def get_batch_data(x, y, batch_size):

    n = len(x)
    i = 0
    while i + batch_size < n:
        yield x[i:i+batch_size], y[i:i+batch_size]
        i += batch_size
    else:
        yield x[i:], y[i:]


def regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:

    m, b = np.random.rand(), np.random.rand()
    m_record, b_record = np.zeros(num_iterations + 1), np.zeros(num_iterations + 1)
    m_record[0] = m
    b_record[0] = b
    for i in range(num_iterations):
        for batch_x, batch_y in get_batch_data(x, y, batch_size):
            y_hat = batch_x * m + b
            loss = np.sum((y_hat - batch_y) ** 2.) / batch_size
            m_g = np.sum(2 * batch_x * (y_hat - batch_y)) / batch_size      # 2x(mx+b-y)
            b_g = np.sum(y_hat - batch_y) / batch_size                # (mx+b-y)

            m -= learning_rate * m_g
            b -= learning_rate * b_g

            print(loss)

        m_record[i+1] = m
        b_record[i+1] = b

    # print(m, b)
    return (m_record, b_record)
