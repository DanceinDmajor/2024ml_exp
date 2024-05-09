import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 读取.mat文件
data = loadmat('exp2_1.mat')

# 提取数据
X = data['X']
Y = data['y']


# 定义线性判别函数分类算法
def linear_discriminant_analysis(X, Y):
    # 计算类别为1的样本均值向量
    mu1 = np.mean(X[Y.flatten() == 1], axis=0)

    # 计算类别为0的样本均值向量
    mu0 = np.mean(X[Y.flatten() == 0], axis=0)

    # 计算类内散度矩阵
    S1 = np.cov(X[Y.flatten() == 1], rowvar=False)
    S0 = np.cov(X[Y.flatten() == 0], rowvar=False)
    Sw = S1 + S0

    # 计算权重向量
    w = np.dot(np.linalg.inv(Sw), (mu1 - mu0))

    # 计算分类阈值
    w0 = -0.5 * (np.dot(mu1, np.dot(np.linalg.inv(Sw), mu1)) - np.dot(mu0, np.dot(np.linalg.inv(Sw), mu0)))

    return w, w0


# 分类算法实现
def classify(X, w, w0):
    return np.dot(X, w) + w0


# 训练线性判别函数模型
w, w0 = linear_discriminant_analysis(X, Y)

# 可视化分类结果
x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = classify(np.c_[X1.ravel(), X2.ravel()], w, w0).reshape(X1.shape)

plt.contourf(X1, X2, Z, levels=[-1, 0, 1], colors=('skyblue', 'salmon'), alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('Linear Discriminant Analysis')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Class')

# 设置坐标轴范围
plt.xlim(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5)
plt.ylim(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5)

plt.savefig('2_1.png')
plt.show()
