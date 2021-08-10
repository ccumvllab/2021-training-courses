from typing import List, Union

import numpy as np


class Variable:

    def __init__(self, value=None):
        self.value = value
        self.grad = None


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
