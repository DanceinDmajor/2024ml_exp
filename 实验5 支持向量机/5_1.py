import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 读取.mat文件
data = loadmat('ex5data1.mat')

# 提取数据
X = data['X']
Y = data['y'].flatten()

# 将Y中的0替换为-1
Y = Y.astype(np.int8)  # 确保Y是整数类型
Y[Y == 0] = -1

# 数据标准化
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# 给数据添加一列常数项1，用于计算偏置b
X = np.hstack((X, np.ones((X.shape[0], 1))))


# 定义线性SVM的损失函数
def hinge_loss(W, X, Y, C):
    distances = 1 - Y * np.dot(X, W)
    distances[distances < 0] = 0  # max(0, distance)
    hinge_loss = C * (np.sum(distances))
    return 0.5 * np.dot(W, W) + hinge_loss


# 定义线性SVM的梯度
def hinge_loss_gradient(W, X, Y, C):
    distances = 1 - Y * np.dot(X, W)
    dw = np.zeros(len(W))
    for i, d in enumerate(distances):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (C * Y[i] * X[i])
        dw += di
    dw = dw / len(Y)  # average
    return dw


# 梯度下降法训练线性SVM
def gradient_descent(X, Y, C=1, learning_rate=0.001, max_epochs=1000):
    W = np.zeros(X.shape[1])
    for epoch in range(max_epochs):
        grad = hinge_loss_gradient(W, X, Y, C)
        W = W - learning_rate * grad
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {hinge_loss(W, X, Y, C)}')
    return W


# 训练模型
W = gradient_descent(X, Y, C=81, learning_rate=0.30, max_epochs=1000)


# 可视化划分边界
def plot_decision_boundary(W, X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', alpha=0.7)

    # 绘制决策边界
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = -(W[0] * x1 + W[2]) / W[1]
    plt.plot(x1, x2, color='k')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Linear SVM Decision Boundary')
    plt.show()


# 创建可视化
plot_decision_boundary(W, X, Y)
