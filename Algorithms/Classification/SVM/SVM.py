import numpy as np  # 导入numpy用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from matplotlib.font_manager import FontProperties  # 导入字体管理用于中文显示

font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)  # 设置中文字体


def load_data(train_file='svm_data_train.txt', test_file='svm_data_test.txt'):
    """
    加载训练集和测试集数据
    """
    train_data = np.loadtxt(train_file, delimiter=',')  # 加载训练集数据
    test_data = np.loadtxt(test_file, delimiter=',')    # 加载测试集数据
    X_train, y_train = train_data[:, :2], train_data[:, 2]  # 前两列为特征，最后一列为标签
    X_test, y_test = test_data[:, :2], test_data[:, 2]      # 同上
    return X_train, y_train, X_test, y_test  # 返回训练集和测试集


def standardize(X_train, X_test):
    """
    对特征进行标准化处理，使均值为0，方差为1
    """
    mean = X_train.mean(axis=0)  # 计算训练集每个特征的均值
    std = X_train.std(axis=0)    # 计算训练集每个特征的标准差
    X_train_std = (X_train - mean) / std  # 标准化训练集
    X_test_std = (X_test - mean) / std    # 标准化测试集（用训练集的均值和方差）
    return X_train_std, X_test_std        # 返回标准化后的数据


def plot_data(X, y, title="Data"):
    """
    绘制数据分布图
    """
    plt.figure(figsize=(8, 8))  # 创建画布
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='+1')    # 绘制正类点
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='-1') # 绘制负类点
    plt.legend()  # 显示图例
    plt.title(title, fontproperties=font)  # 设置标题
    plt.xlabel("Feature 1")  # x轴标签
    plt.ylabel("Feature 2")  # y轴标签
    plt.show()  # 显示图像


def gaussian_kernel(x1, x2, sigma=0.5):
    """
    高斯核函数（RBF核），用于计算样本之间的非线性相似度
    K(x1, x2) = exp(-||x1-x2||^2 / (2*sigma^2))
    """
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))  # 返回高斯核值


def compute_kernel_matrix(X, sigma=0.5):
    """
    计算所有样本之间的核矩阵K
    K[i, j] = gaussian_kernel(X[i], X[j])
    """
    m = X.shape[0]  # 样本数量
    K = np.zeros((m, m))  # 初始化核矩阵
    for i in range(m):    # 遍历每个样本
        for j in range(m):  # 遍历每个样本
            K[i, j] = gaussian_kernel(X[i], X[j], sigma)  # 计算核函数
    return K  # 返回核矩阵


def train_svm(X, y, C=1.0, sigma=0.5, lr=0.01, num_iters=200):
    """
    SVM训练主流程（核方法，梯度下降近似，实际SVM应用SMO算法）
    目标：max ∑α_i - 0.5∑∑α_iα_jy_iy_jK(x_i,x_j)
    约束：0<=α_i<=C, ∑α_iy_i=0
    这里只做简化近似，实际应用请用libsvm等库
    """
    m = X.shape[0]  # 样本数量
    alpha = np.zeros(m)  # 初始化拉格朗日乘子α
    K = compute_kernel_matrix(X, sigma)  # 计算核矩阵
    for it in range(num_iters):  # 迭代训练
        # 梯度近似更新alpha
        grad = 1 - y * (K @ (alpha * y))  # 计算梯度（简化版）
        alpha += lr * grad                # 梯度下降更新α
        alpha = np.clip(alpha, 0, C)      # 保证α在[0, C]范围内
        if it % 50 == 0 or it == num_iters - 1:  # 每隔50步打印一次
            print(f"Iter {it}: mean alpha={np.mean(alpha):.4f}")  # 打印平均α
    return alpha, K  # 返回训练好的α和核矩阵


def predict_svm(X_train, y_train, alpha, X_test, sigma=0.5):
    """
    SVM预测函数
    对每个测试样本x，计算f(x) = ∑α_i y_i K(x_i, x)
    sign(f(x))为预测类别
    """
    y_pred = []  # 存储预测结果
    for x in X_test:  # 遍历每个测试样本
        s = 0  # 初始化决策值
        for i in range(len(X_train)):  # 遍历每个训练样本
            s += alpha[i] * y_train[i] * gaussian_kernel(X_train[i], x, sigma)  # 累加支持向量贡献
        y_pred.append(np.sign(s))  # sign为类别
    return np.array(y_pred)  # 返回预测结果


def plot_svm_decision_boundary(X_train, y_train, alpha, sigma=0.5, title="SVM决策边界"):
    """
    绘制SVM高斯核决策边界
    方法：对网格点进行预测，画出f(x)=0的等高线
    """
    # 生成网格点
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5  # x轴范围
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5  # y轴范围
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))  # 网格
    grid = np.c_[xx.ravel(), yy.ravel()]  # 网格点坐标
    y_grid = predict_svm(X_train, y_train, alpha, grid, sigma)  # 网格点预测
    y_grid = y_grid.reshape(xx.shape)  # 变为网格形状

    plt.figure(figsize=(8, 8))  # 创建画布
    plt.contour(xx, yy, y_grid, levels=[0], linewidths=2.0, colors='g')  # 绘制决策边界
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='+1')    # 正类点
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], c='blue', label='-1') # 负类点
    plt.title(title, fontproperties=font)  # 标题
    plt.xlabel("Feature 1")  # x轴标签
    plt.ylabel("Feature 2")  # y轴标签
    plt.legend()  # 图例
    plt.show()    # 显示图像


def SVM():
    # 1. 数据提取
    X_train, y_train, X_test, y_test = load_data()  # 加载数据
    plot_data(X_train, y_train, "训练集分布")         # 绘制训练集分布
    plot_data(X_test, y_test, "测试集分布")           # 绘制测试集分布

    # 2. 数据处理
    X_train_std, X_test_std = standardize(X_train, X_test)  # 标准化数据

    # 3. 训练
    alpha, K = train_svm(X_train_std, y_train, C=1.0, sigma=0.5, lr=0.01, num_iters=300)  # SVM训练

    # 4. 训练后可视化（预测结果）
    y_pred_train = predict_svm(X_train_std, y_train, alpha, X_train_std)  # 训练集预测
    y_pred_test = predict_svm(X_train_std, y_train, alpha, X_test_std)    # 测试集预测

    print(f"训练集准确率: {np.mean(y_pred_train == y_train) * 100:.2f}%")  # 打印训练集准确率
    print(f"测试集准确率: {np.mean(y_pred_test == y_test) * 100:.2f}%")    # 打印测试集准确率

    plot_data(X_train_std, y_pred_train, "训练集预测分布")  # 绘制训练集预测分布
    plot_data(X_test_std, y_pred_test, "测试集预测分布")    # 绘制测试集预测分布
    plot_svm_decision_boundary(X_train_std, y_train, alpha, sigma=0.5, title="SVM决策边界")  # 绘制决策边界


if __name__ == '__main__':
    SVM()  # 主函数入口，执行SVM流程
