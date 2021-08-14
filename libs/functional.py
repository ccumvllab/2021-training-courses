from typing import List, Union, Tuple

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


def regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(num_samples)
    np.random.shuffle(idx)

    m = np.zeros(num_iterations + 1)
    b = np.zeros(num_iterations + 1)

    m[0] = np.random.normal()
    b[0] = np.random.normal()

    for i in range(num_iterations):
        j = slice(i * batch_size % num_samples, (i + 1) * batch_size % num_samples)
        x_in_batch = x[j]
        y_in_batch = y[j]

        # Compute the gradients
        m_grad = np.sum(2 * x_in_batch * (m[i] * x_in_batch + b[i] - y_in_batch)) / num_samples
        b_grad = np.sum(2 * (m[i] * x_in_batch + b[i] - y_in_batch)) / num_samples

        # Update m and b
        m[i + 1] = m[i] - learning_rate * m_grad
        b[i + 1] = b[i] - learning_rate * b_grad

    return m, b
