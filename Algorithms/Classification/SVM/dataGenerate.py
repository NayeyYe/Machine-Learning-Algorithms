import numpy as np
import matplotlib.pyplot as plt

def generate_svm_data(n_samples=500, random_state=42, noise_scale=1.0, boundary_soft_ratio=0.05):
    """
    生成用于SVM训练的软间隔二维数据集（不划分训练测试集）
    仅在边界附近少量点做混合和噪声，其余区域分界明显
    """
    np.random.seed(random_state)
    center_pos = [2, 2]
    center_neg = [-2, -2]
    X_pos = np.random.randn(n_samples // 2, 2) * noise_scale + center_pos
    X_neg = np.random.randn(n_samples // 2, 2) * noise_scale + center_neg
    y = np.hstack((np.ones(n_samples // 2), -np.ones(n_samples // 2)))
    X = np.vstack((X_pos, X_neg))

    # 仅在边界附近做少量混合
    dist = np.linalg.norm(X, axis=1)
    boundary_num = int(n_samples * boundary_soft_ratio)
    boundary_indices = np.argsort(np.abs(dist))[:boundary_num]
    X[boundary_indices] += np.random.randn(boundary_num, 2) * (noise_scale * 1.2)
    y[boundary_indices] *= -1

    # 打乱数据
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    return X, y


def train_data_split(X, y, train_ratio=0.7):
    """
    按比例划分训练集和测试集
    """
    n_samples = X.shape[0]
    split = int(n_samples * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test


def plot_svm_data(X_train, y_train, X_test, y_test):
    """
    分别绘制训练集和测试集分布
    """


    # 绘制训练集
    plt.figure(figsize=(8, 8))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Train +1')
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], c='blue', label='Train -1')
    plt.legend()
    plt.title("SVM Soft Margin Training Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # 绘制测试集
    plt.figure(figsize=(8, 8))
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='orange', marker='x', label='Test +1')
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], c='cyan', marker='x', label='Test -1')
    plt.legend()
    plt.title("SVM Soft Margin Test Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def save_svm_data(X_train, y_train, X_test, y_test, prefix='svm_data'):
    """
    保存训练集和测试集到文件
    """
    np.savetxt(f"{prefix}_train.txt", np.hstack((X_train, y_train.reshape(-1, 1))), delimiter=',', comments='')
    np.savetxt(f"{prefix}_test.txt", np.hstack((X_test, y_test.reshape(-1, 1))), delimiter=',', comments='')


if __name__ == '__main__':
    X, y = generate_svm_data()
    X_train, y_train, X_test, y_test = train_data_split(X, y)
    plot_svm_data(X_train, y_train, X_test, y_test)
    save_svm_data(X_train, y_train, X_test, y_test)
    print("SVM train/test data generated and saved.")
