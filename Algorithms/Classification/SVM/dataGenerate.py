import numpy as np
import matplotlib.pyplot as plt

def generate_svm_data(n_samples=1000, random_state=42, inner_radius=1.0, outer_radius=1.6, noise_scale=0.18, cross_ratio=0.20):
    """
    生成用于SVM高斯核训练的同心环形数据集（不划分训练测试集）
    数据分布无法被线性超平面分割，边界更soft，交叉更多，适合核方法
    """
    np.random.seed(random_state)
    # 正类：内环
    theta_pos = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r_pos = np.random.normal(inner_radius, noise_scale, n_samples // 2)
    X_pos = np.column_stack((r_pos * np.cos(theta_pos), r_pos * np.sin(theta_pos)))
    y_pos = np.ones(n_samples // 2)

    # 负类：外环
    theta_neg = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r_neg = np.random.normal(outer_radius, noise_scale, n_samples // 2)
    X_neg = np.column_stack((r_neg * np.cos(theta_neg), r_neg * np.sin(theta_neg)))
    y_neg = -np.ones(n_samples // 2)

    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    # 在边界附近制造更多交叉
    dist = np.linalg.norm(X, axis=1)
    # 边界定义为距离在[inner_radius+0.5, outer_radius-0.5]之间
    boundary_mask = (dist > inner_radius + 0.5) & (dist < outer_radius - 0.5)
    boundary_indices = np.where(boundary_mask)[0]
    n_cross = int(n_samples * cross_ratio)
    if len(boundary_indices) > 0:
        cross_indices = np.random.choice(boundary_indices, min(n_cross, len(boundary_indices)), replace=False)
        y[cross_indices] *= -1
        X[cross_indices] += np.random.randn(len(cross_indices), 2) * (noise_scale * 2.5)

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
    plt.title("SVM Gaussian Kernel Training Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # 绘制测试集
    plt.figure(figsize=(8, 8))
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='orange', marker='x', label='Test +1')
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], c='cyan', marker='x', label='Test -1')
    plt.legend()
    plt.title("SVM Gaussian Kernel Test Data")
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
