import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist

# 读取.mat文件
data = loadmat('ex5data2.mat')

# 提取数据
X = data['X']
Y = data['y'].flatten()

# 将Y中的0替换为-1
Y = Y.astype(np.int8)  # 确保Y是整数类型
Y[Y == 0] = -1

# 定义高斯核函数
def gaussian_kernel(X1, X2, sigma=1.0):
    if X1.ndim == 1:
        X1 = X1[np.newaxis, :]
    if X2.ndim == 1:
        X2 = X2[np.newaxis, :]
    return np.exp(-cdist(X1, X2, 'sqeuclidean') / (2 * sigma ** 2))

# 定义非线性SVM的损失函数
def hinge_loss(K, Y, alpha, C):
    distances = 1 - Y * np.dot(K, alpha)
    distances[distances < 0] = 0  # max(0, distance)
    hinge_loss = C * (np.sum(distances))
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) + hinge_loss

# 定义非线性SVM的梯度
def hinge_loss_gradient(K, Y, alpha, C):
    distances = 1 - Y * np.dot(K, alpha)
    d_alpha = np.zeros(len(alpha))
    for i, d in enumerate(distances):
        if max(0, d) == 0:
            d_alpha += alpha
        else:
            d_alpha += alpha - (C * Y[i] * K[i])
    return d_alpha / len(Y)  # Average gradient

# 梯度下降法训练非线性SVM
def gradient_descent(K, Y, C=1, learning_rate=0.001, max_epochs=1000):
    alpha = np.zeros(K.shape[0])
    for epoch in range(max_epochs):
        grad = hinge_loss_gradient(K, Y, alpha, C)
        alpha = alpha - learning_rate * grad
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {hinge_loss(K, Y, alpha, C)}')
    return alpha

# 计算高斯核矩阵
sigma = 0.05
K = gaussian_kernel(X, X, sigma)

# 训练模型
alpha = gradient_descent(K, Y, C=20, learning_rate=0.01, max_epochs=3000)

# 定义预测函数
def predict(X_train, X_test, alpha, sigma=0.1):
    K_test = gaussian_kernel(X_test, X_train, sigma)
    return np.sign(np.dot(K_test, alpha))

# 可视化划分边界
def plot_decision_boundary(X, Y, alpha, sigma=0.1):
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', alpha=0.7)

    # 绘制决策边界
    x1, x2 = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100),
                         np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
    Z = predict(X, np.c_[x1.ravel(), x2.ravel()], alpha, sigma)
    Z = Z.reshape(x1.shape)
    plt.contour(x1, x2, Z, levels=[0], colors='k')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gaussian Kernel SVM Decision Boundary')
    plt.show()

# 创建可视化
plot_decision_boundary(X, Y, alpha, sigma)
