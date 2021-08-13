#pip3 install PyQt5==5.9.2

import numpy, matplotlib
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

def plot_polynomial(polynomial_func,x_val): #adapt from stackoverflow
    plt.figure(figsize=(8,4),dpi=150)

    #apply x to polynomial function
    p = Polynomial(polynomial_func)
    print(p)
    d = (p.deriv())
    print(d)

    s1,s2 = p(x_val),d(x_val)
    print(s1,s2)

    window_size = 20

    x = []
    y = []
    for i in numpy.arange((x_val-window_size),(x_val+window_size),0.1): #axes range on plot (when degree increase --> overflow/underflow)
        x.append(i)
        y.append(p(i))

    y_s,y_e = (-window_size) * s2 + s1, (window_size)* s2 + s1

    plt.plot([x_val-window_size,x_val+window_size],[y_s,y_e])
    plt.plot(x,y)
    plt.scatter(x_val,p(x_val), s=5, c='r')
    plt.show()
    plt.close()
