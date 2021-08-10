import matplotlib
import matplotlib.pyplot as plt

def plot_polynomial(x_cut, a: List = None):
    f = Polynomial(a)
    x=np.linspace(x_cut-10 ,x_cut+10 ,1024) # x軸 (x_start, x_stop, y_tall)
    y=f(x).value  # y軸
    
    plt.plot(x,y,color='blue',label='polynomial line')  # function graph
    plt.plot(x, (x-x_cut) *f(x_cut).grad +f(x_cut).value ,color='red',label='tangent line') # tangent line
    
    plt.legend(loc='best',title='Polynomial graph_fanxinyun') # 標籤放置
    plt.savefig('plot_polynomial.png') #在show之前要先save
    plt.show()   #show圖
