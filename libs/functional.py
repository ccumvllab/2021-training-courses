from typing import List, Union,Tuple
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

    def get_batch(self, x,y,batch_size,num_samples): #將資料重新shape
        return np.reshape(x,(int(num_samples/batch_size),batch_size)),np.reshape(y,(int(num_samples/batch_size),batch_size))

    def regression_sgd(self,x, y, num_samples, num_iterations, batch_size, learning_rate) -> Tuple[np.ndarray, np.ndarray]:
        m, b = np.random.rand(), np.random.rand()
        m_history = [m]
        b_history = [b]
        x,y = self.get_batch(x,y,batch_size,num_samples)

        # loss function定義為-> sigmod(mx+b-y)^2
        for i in range(num_iterations):
            m_grad = 0.0
            b_grad = 0.0
            for j in range(int(num_samples/batch_size)):
                y_hat = m * x[j] + b #y_hat
                b_grad = np.sum(2.0 * (y_hat -y[j]) * 1.0)/batch_size #loss function對b做偏微分=>2*(mx+b-y)*1
                m_grad = np.sum(2.0 * (y_hat -y[j]) * x[j])/batch_size #loss function對m做偏微分=>2*(mx+b-y)*x

            m = m - learning_rate * m_grad
            b = b - learning_rate * b_grad
            m_history.append(m)
            b_history.append(b)
        return  np.array(m_history),np.array(b_history)


