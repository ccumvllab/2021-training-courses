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
        f_value=0
        f_grad=0
        # 計算function
        for i in range(self.a.size):
          f_value += self.a[i] * x**i
          
        # 計算微分
        for i in range(self.a.size-1):
          f_grad += self.a[i+1] *(i+1) * x**i

        # 計算結果儲存至Variable  
        function=Variable(f_value,f_grad)
        return function

from typing import Tuple

def shuffle_data(x, y):
    assert len(x) == len(y)
    # 合併 x , y
    training_data = np.vstack((x, y))
    # 轉向之後 shuffle 不會打散成對資料
    training_data = training_data.T
    # shuffle
    np.random.shuffle(training_data)
    training_data = training_data.T
    X = training_data[0, :]
    Y = training_data[1, :]
    return X, Y


def regression_sgd(x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:
    m, b = np.random.randn(), np.random.randn()  # 隨機初始化 m, b
    m_i, b_i = np.zeros(num_iterations + 1), np.zeros(num_iterations + 1)
    # 儲存初始化參數
    m_i[0] = m
    b_i[0] = b
    # 做幾輪 epoch
    for i in range(num_iterations):
        # Shuffle
        x, y = shuffle_data(x, y)
        for start in range(0, num_samples, batch_size):
            # 取 Batch 資料
            stop = start + batch_size
            if stop <= num_iterations:
                x_batch_data, y_batch_data = x[start:stop], y[start:stop]
            else:
                x_batch_data, y_batch_data = x[start:num_samples], y[start:num_samples]

            y_exp = m * x_batch_data + b
            # MSE Loss
            MSE = np.sum((y_exp - y_batch_data) ** 2) / batch_size
            # 參數的 gradients
            m_grad = np.sum(2 * x_batch_data *
                            (y_exp - y_batch_data)) / batch_size
            b_grad = np.sum(y_exp - y_batch_data) / batch_size
            # 每過一個 Batch 更新一次參數
            m = m - m_grad * learning_rate
            b = b - b_grad * learning_rate
        # 每次 epoch 儲存一次參數
        m_i[i+1] = m
        b_i[i+1] = b
    return (m_i, b_i)
