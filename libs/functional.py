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

        # Mini-batch stochastic gradient descent
        # 計算這個batch中的單一sample之 loss function -> 藉由這個loss計算gradient -> 更新參數 -> next sample in bacth
        # 直到這個batch中的每個sample都更新過參數 -> next iteration
        '''
        for j in range(batch_size):
            m[i+1] -= learning_rate*batch_x[j]*(m[i+1]*batch_x[j]+-batch_y[j])/batch_size # 2x(mx+b-y)
            b[i+1] -= learning_rate*(m[i+1]*batch_x[j]+-batch_y[j])/batch_size #2(mx+b-y)
        '''

        # Stochastic mini-batch gradient descent
        # 計算這個batch的loss function -> 藉由這個loss計算gradient -> 更新參數 -> next iteration
        b_temp = np.full([1,100],b[i]) #建立一個1*100的一維陣列b[i]，方便y_hat的計算
        y_hat = m[i] * x + b_temp
        m_gt = np.sum(2 * x * (y_hat - y))/batch_size # m 的 gradient計算
        b_gt = np.sum(2 * (y_hat - y))/batch_size # b 的 gradient計算
        m[i+1] -= learning_rate * m_gt # update m gradient
        b[i+1] -= learning_rate * b_gt # update b gradient

    return (m,b)

