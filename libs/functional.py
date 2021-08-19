#https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843
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
    m,b = np.zeros(num_iterations+1), np.zeros(num_iterations+1) #array saving updated m,b
    m[0],b[0] = np.random.randn(),np.random.randn() #initial random value data points
    samples  = list(zip(x,y))
    np.random.shuffle(samples)
    for i in range(num_iterations):
        temp_m,temp_b = m[i],b[i]
        for j in range(0,num_samples,batch_size):
            x0 = np.array([x for x,_ in samples[j:j+batch_size]]) #samples in x
            y0 = np.array([y for _,y in samples[j:j+batch_size]]) #samples in y
            y1 = temp_m * x0 + temp_b  #y-hat = mx+b
            m_grad = np.sum(2*x0*(y1-y0))/batch_size #(2x(mx+b-y))/batch_size = (2x(y1-y0))/batch_size
            b_grad = np.sum(2*(y1-y0))/batch_size #(2(mx+b-y))/batch_size = (2(y1-y0))/batch_size
            temp_m -= learning_rate*m_grad
            temp_b -= learning_rate*b_grad
        m[i+1] = temp_m
        b[i+1] = temp_b
    return (m,b)
