import numpy as np

def generate_data():
    """
    生成用于核函数逻辑回归的椭圆内外数据集
    """
    np.random.seed(0)
    m = 500  # 样本数量

    # 椭圆参数
    a, b = 3, 1.5  # 长轴和短轴

    # 生成正类（椭圆内）
    theta_pos = np.random.uniform(0, 2 * np.pi, m // 2)
    r_pos = np.sqrt(np.random.uniform(0, 1, m // 2))
    X1 = np.column_stack((a * r_pos * np.cos(theta_pos), b * r_pos * np.sin(theta_pos)))
    y1 = np.ones((m // 2, 1))

    # 生成负类（椭圆外，半径在[1.2, 2]之间）
    theta_neg = np.random.uniform(0, 2 * np.pi, m // 2)
    r_neg = np.sqrt(np.random.uniform(1.2, 2, m // 2))
    X2 = np.column_stack((a * r_neg * np.cos(theta_neg), b * r_neg * np.sin(theta_neg)))
    y2 = np.zeros((m // 2, 1))

    # 合并
    X = np.vstack((X1, X2))
    y = np.vstack((y1, y2))

    return X, y


def plot_data(X, y):
    """
    绘制数据分布图
    """
    import matplotlib.pyplot as plt

    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置

    plt.figure(figsize=(10, 10))
    plt.scatter(X[pos, 0], X[pos, 1], c='red', label='Positive Class')  # 正类用红色
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', label='Negative Class')  # 负类用蓝色
    plt.title("Data Distribution")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def save_data(X, y, filename = 'data.txt'):
    """
    保存数据到文件
    """
    data = np.hstack((X, y))
    np.savetxt(filename, data, delimiter=',' , comments='')


if __name__ == '__main__':
    X, y = generate_data()  # 生成数据
    plot_data(X, y)  # 绘制数据分布图
    save_data(X, y)  # 保存数据到文件
    print("Data generated and plotted successfully.")