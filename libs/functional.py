from typing import List, Union

import numpy as np
import pysnooper


class Variable:

    def __init__(self, value=None,grad = None):
        self.value = value
        self.grad = grad

#@pysnooper.snoop()
class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        self.x = x
        total = 0
        grad = 0
        coef = self.a
        exp = [i for i in range(len(self.a))]
        for i,(k,v) in enumerate(zip(coef,exp)):
            total += k * self.x ** v
            if i != 0:
                grad += k*v * self.x ** (v-1)
        return Variable(total,grad)


# 4+2x+6x^2+9x^3+x^4 -> 940
# 2+12x+27x^2+4x^3 -> 738
