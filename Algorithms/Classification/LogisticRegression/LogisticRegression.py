import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)


# 加载数据函数
def load_data():
    # 用np.loadtxt加载txt文件，如果是其他文件，再说吧
    # 加载一下数据
    filename = 'data.txt'
    data = np.loadtxt(filename, delimiter=',', dtype=np.float64)
    # print(data.shape) # (500, 3)
    # 前两列为坐标，最后一列为标签
    X = data[:, 0:-1]  # (500, 2)
    y = data[:, -1]  # (500, 1)

    return X, y


# 处理数据
def data_processing(X, y):
    # 这里对标签不进行处理
    # 将X = (X1, X2)进行处理
    X = mapFeature(X[:, 0], X[:, 1])
    # X_out = (1, x1, x2, x1^2, x1*x2, x2^2)
    # 此时X_out的形状为 (500, 6)
    return X


def mapFeature(X1, X2):
    # 映射的最高次方
    degree = 2
    # 映射结果，用于取代 X
    X_out = np.ones(shape=(X1.shape[0], 1))
    '''
    映射为 1,x1,x2,x1^2,x1,x2,x2^2 多项式组合
    X_out = (1, x1, x2, x1^2, x1*x2, x2^2) 
            = (1, x1^i * x2^j) for i + j == degree+1
    循环i,j 然后按列拼接即可
    '''
    for i in range(1, degree + 1):
        for j in range(i + 1):
            # 计算 x1^i * x2^j
            temp = (X1 ** (i - j)) * (X2 ** j)
            # 拼接到 X_out
            X_out = np.hstack((X_out, temp.reshape(-1, 1)))
    return X_out


def sigmoid(z):
    # Sigmoid函数
    h = np.zeros((len(z), 1))  # 初始化h为零向量
    h = 1 / (1 + np.exp(-z))  # Sigmoid函数公式
    return h


def gradient(theta, X, y, _lambda):
    m = X.shape[0]  # 样本数量
    grad = np.zeros((theta.shape[0], 1))  # 初始化梯度为零向量

    h = sigmoid(np.dot(X, theta))  # 计算h(z)
    '''
    注意:
    在逻辑回归中，我们的目标是获取一个线性的组合
        result = beta_0 + beta_1 * x1 + beta_2 * x2 + ... + beta_n * xn
    写成矩阵形式为 result = (1, x1, x2, ..., xn) * (beta_0, beta_1, beta_2, ..., beta_n)^T
    我们这个的theta就是 (beta_0, beta_1, beta_2, ..., beta_n)^T,
    但是在正则化过程中, 第一个是beta_0, 也就是theta[0]不参与正则化
    所以我们要复制一份theta, 然后将第一个元素置为0
    '''
    theta1 = theta.copy()  # 复制一份theta
    theta1[0] = 0  # 将第一个元素置为0, 因为正则化不包含theta[0]

    # 计算梯度
    grad = np.dot(np.transpose(X), h - y) / m + _lambda / m * theta1  # 正则化的梯度
    return grad  # 返回梯度向量


# 在给定的theta和X下，计算loss
# 这是我们需要优化的函数
def lossFunction(theta, X, y, _lambda):
    # 初始参数
    m = X.shape[0]  # 样本数量
    loss = 0  # loss初始化为0

    h = sigmoid(np.dot(X, theta))  # 计算h(z)
    '''
    注意:
    在logistic回归中，我们的目标是获取一个线性的组合
        result = beta_0 + beta_1 * x1 + beta_2 * x2 + ... + beta_n * xn
    写成矩阵形式为 result = (1, x1, x2, ..., xn) * (beta_0, beta_1, beta_2, ..., beta_n)^T
    我们这个的theta就是 (beta_0, beta_1, beta_2, ..., beta_n)^T,
    但是在正则化过程中, 第一个是beta_0, 也就是theta[0]不参与正则化
    所以我们要复制一份theta, 然后将第一个元素置为0
    '''

    theta1 = theta.copy()  # 复制一份theta
    theta1[0] = 0  # 将第一个元素置为0, 因为正则化不包含theta[0]

    temp = np.dot(np.transpose(theta1), theta1)  # 计算theta的平方和
    # 计算代价函数
    loss = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y), np.log(1 - h)) + temp * _lambda / 2) / m
    return loss


def plot_decision_boundary(theta, X, y):
    # 绘制数据点
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.figure(figsize=(10, 10))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')

    # 生成网格
    u = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
    v = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            mapped = mapFeature(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = np.dot(mapped, theta).item()

    z = z.T
    plt.contour(u, v, z, levels=[0], linewidths=2.0, colors='g')
    plt.title(u'决策边界', fontproperties=font)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def predict(X, theta):
    # 预测函数
    # 计算预测值
    h = sigmoid(np.dot(X, theta))  # 计算h(z)
    # 将预测值转换为0或1
    p = (h >= 0.5).astype(int)  # 如果h >= 0.5，则预测为1，否则为0
    return p  # 返回预测结果


def train(theta, X, y, _lambda, lr=0.1, num_iters=1000, tol=1e-6, verbose=False):
    """
    使用批量梯度下降法训练逻辑回归参数
    :param theta: 初始参数 (n, 1)
    :param X: 特征矩阵 (m, n)
    :param y: 标签向量 (m, 1) 或 (m,)
    :param _lambda: 正则化参数
    :param lr: 学习率
    :param num_iters: 最大迭代次数
    :param tol: 收敛阈值
    :param verbose: 是否打印loss
    :return: 训练好的参数theta
    """

    theta = theta.copy()
    y = y.reshape(-1, 1)
    prev_loss = lossFunction(theta, X, y, _lambda)
    for i in range(num_iters):
        # 计算梯度
        grad = gradient(theta, X, y, _lambda)
        # 更新参数
        theta = theta - lr * grad
        # 计算当前的loss
        curr_loss = lossFunction(theta, X, y, _lambda)
        # 打印当前的loss
        if verbose and (i % 100 == 0 or i == num_iters - 1):
            print(f"Iter {i}: loss = {curr_loss}")
        # 检查收敛条件
        if np.abs(prev_loss - curr_loss) < tol:
            if verbose:
                print(f"Converged at iter {i}")
            break
        # 更新前一个loss
        prev_loss = curr_loss
    return theta


# 这是主函数入口
def LogisticRegression():
    # 加载数据
    X, y = load_data()
    plot_data(X, y)

    # 开始处理数据
    # 这里的数据处理是，将X = (X1, X2)这个二维数据变成多项式(1, X1, X2, X1^2, X1X2, X2^2)
    X = data_processing(X, y)  # (500, 6)

    # 逻辑回归中的各种参数初始化
    # 第一个是参数向量theta，第二个是正则化参数lambda
    '''
    在逻辑回归中，theta是参数向量，通常初始化为零向量
    这里的X.shape[1]是特征的数量，theta的形状为 (n, 1)，其中n是特征的数量
    这里的X.shape[1] = 6，所以theta的形状为 (6, 1)
    '''
    theta = np.zeros((X.shape[1], 1))
    # 正则化参数
    _lambda = 0.1
    '''
    在逻辑回归中，正则化参数lambda用于控制模型的复杂度
    这里设置为1.0是一个常见的初始值，可以根据需要进行调整
    '''

    # 代价函数, 也就是loss, 我们用负对数极大似然估计函数用于计算代价
    loss = lossFunction(theta, X, y, _lambda)
    print(f'Initial loss: {loss}')  # 打印初始代价

    # result = optimize.fmin_bfgs(lossFunction, theta, fprime=gradient, args=(X, y, _lambda))
    result = train(theta, X, y, _lambda, lr=0.1, num_iters=2000, tol=1e-7, verbose=True)
    p = predict(X, result)  # 预测
    # 修正准确率计算，确保p和y都是一维向量
    acc = np.mean(p.reshape(-1) == y.reshape(-1)) * 100
    print(f'在训练集上的准确度为 {acc:.2f}%')

    X = X[:, 1:]  # 去掉第一列的1
    y = y.reshape(-1, 1)  # 将y转换为列向量
    plot_decision_boundary(result, X, y)  # 画决策边界


# 这个函数是用来查看数据分布的样子的
def plot_data(X, y):
    """
    绘制数据分布图
    :param X: 特征矩阵 (m, 2)
    :param y: 标签向量 (m, 1) 或 (m,)
    """
    # 找到y == 1和y == 0的坐标
    positive_data = np.where(y == 1)
    negative_data = np.where(y == 0)

    # 作图
    plt.figure(figsize=(10, 10))
    plt.plot(X[positive_data, 0], X[positive_data, 1], 'ro', label='Positive example')
    plt.plot(X[negative_data, 0], X[negative_data, 1], 'bo', label='Negative example')
    plt.title(u'散点图', fontproperties=font)
    plt.show()


if __name__ == '__main__':
    LogisticRegression()  # 调用逻辑回归函数
