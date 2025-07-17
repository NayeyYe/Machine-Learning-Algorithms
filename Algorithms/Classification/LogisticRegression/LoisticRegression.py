import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r'c:\windows\fonts\simsun.ttc', size = 14)


# 加载数据函数
def load_data():
    # 用np.loadtxt加载txt文件，如果是其他文件，再说吧
    # 加载一下数据
    filename = 'data.txt'
    data = np.loadtxt(filename, delimiter = ',', dtype = np.float64)
    # print(data.shape) # (118, 3)
    # 前两列为坐标，最后一列为标签
    X = data[ : , 0:-1] # (118, 2)
    y = data[ : , -1] # (118, 1)

    return X, y


# 处理数据
def data_processing(X, y):
    # 这里对标签不进行处理
    # 将X = (X1, X2)进行处理
    X = mapFeature(X[:, 0], X[:, 1])
    # X_out = (1, x1, x2, x1^2, x1*x2, x2^2)
    # 此时X_out的形状为 (118, 6)
    return X


def mapFeature(X1, X2):
    # 映射的最高次方
    degree = 2
    # 映射结果，用于取代 X
    X_out = np.ones(shape = (X1.shape[0], 1))
    '''
    映射为 1,x1,x2,x1^2,x1,x2,x2^2 多项式组合
    X_out = (1, x1, x2, x1^2, x1*x2, x2^2) 
            = (1, x1^i * x2^j) for i + j == degree+1
    循环i,j 然后按列拼接即可
    '''
    for i in range(1, degree + 1):
        for j in range(i +1):
            # 计算 x1^i * x2^j
            temp = (X1 ** (i - j)) * (X2 ** j)
            # 拼接到 X_out
            X_out = np.hstack((X_out, temp.reshape(-1, 1)))
    return X_out



# 这是主函数入口
def LogisticRegression():
    # 加载数据
    X, y = load_data()
    # 开始处理数据
    # 这里的数据处理是，将X = (X1, X2)这个二维数据变成多项式(1, X1, X2, X1^2, X1X2, X2^2)
    X = data_processing(X, y) # (118, 6)



# 这个函数是用来查看数据分布的样子的
def plot_data():
    X, y = load_data()
    # 找到y == 1和y == 0的坐标
    positive_data = np.where(y == 1)
    negative_data = np.where(y == 0)

    # 作图
    plt.figure(figsize = (10,10))
    plt.plot(X[positive_data, 0], X[positive_data, 1], 'ro', label = 'Positive example')
    plt.plot(X[negative_data, 0], X[negative_data, 1], 'bo', label = 'Negative example')
    plt.title(u'散点图', fontproperties = font)
    plt.show()

if __name__ == '__main__':
    plot_data()