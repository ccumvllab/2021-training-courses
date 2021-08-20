from typing import List, Union, Tuple

import numpy as np
from random import sample


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
    m, b = np.random.rand(), np.random.rand() # 隨機初始化 m , b
    result_m, result_b = np.empty( 
        num_iterations + 1), np.empty(num_iterations + 1)  # 初始化大小為 num_iterations + 1 的 array
    result_m[0], result_b[0] = m, b
    for i in range(num_iterations):
        batch_x, batch_y = getBatchData(x, y, num_samples, batch_size)
        y_hat = m * batch_x + b
        mseLoss = sum(( y_hat - batch_y ) **2 ) / batch_size # 計算 MSE Loss
        gradient_m = sum( 2 * batch_x * ( y_hat - batch_y ) ) / batch_size # 計算 m的gradients
        gradient_b = sum( 2 * ( y_hat - batch_y ) ) / batch_size # 計算 b的gradients
        m -= learning_rate * gradient_m 
        b -= learning_rate * gradient_b # 更新參數
        result_m[i+1] = m
        result_b[i+1] = b
    return result_m, result_b

# 隨機選取 batch_size 個資料
def getBatchData(x, y, num_samples, batch_size): 
    sampleRange = range(0, num_samples)
    randomIndex = sample(sampleRange, batch_size) 
    batch_x = x[randomIndex]
    batch_y = y[randomIndex]
    return batch_x, batch_y
