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

def plot_regression(file, x, y, prediction, groundtruth):
    # Data 散佈圖
    plt.scatter(x, y, s=10, alpha=0.5, color='green', label='Data points')
    # Real Line
    m_gt, b_gt = groundtruth[0], groundtruth[1]
    plt.plot(x, m_gt*x + b_gt, alpha=0.5, color='red',
             label='Groundtruth', linewidth=1)
    # Predict Line
    m, b = prediction[0], prediction[1]
    plt.plot(x, m[-1] * x + b[-1], alpha=0.5, color='blue',
             label='Prediction', linestyle='--', linewidth=1)

    plt.legend(loc='best')  # 標籤放置
    plt.savefig(file, dpi=300)  # 在show之前要先save 解析度改為300
    plt.show()  # show圖
    
    
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永無BUG
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
