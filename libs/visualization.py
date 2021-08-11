from typing import List
from matplotlib import pyplot as plt
import numpy as np

# 畫多項式函式圖形   b:輸入x座標求切線
def plot_polynomial(a: List = None, b: float = None):
    func = np.poly1d(np.array(a))
    # x 的橫坐標
    x = np.linspace(-10, 10, 30)
    # 得到y的對應值
    y = func(x)
    # 求 x=b之座標 
    x1 = b
    y1 = func(b)
    # 定義x在切線中的範圍
    xrange = np.linspace(x1-5, x1+5, 10) 
    # 求切線斜率
    deriv = func.deriv()
    # 繪圖
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x1, y1, color='red', s=20)
    plt.plot(xrange, targetLine(deriv(x1),xrange, x1, y1), 'C3--', linewidth = 2)
    # 輸出png檔
    plt.savefig('plot_polynomial.png')
    # 顯示函數圖形
    plt.show()
# 切線方程式    
def targetLine(m,x,x1,y1):
    return m*(x-x1) + y1

