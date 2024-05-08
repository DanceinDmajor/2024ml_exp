
#Parzen窗估计
import pandas as pd
import numpy as np
import math

class ParzenClassifier:
    def __init__(self, data_file):
        # 从Excel文件中读取数据，文件的头部是两行
        self.data = pd.read_excel(data_file, header=[0, 1])
        # 分别获取类别1、2、3的数据
        self.trainSet_1 = self.data['类1']
        self.trainSet_2 = self.data['类2']
        self.trainSet_3 = self.data['类3']

    def window(self, sample, trainSample, h):
        # 计算样本和训练样本之间的差值
        diff = sample.values - trainSample.values
        # 计算差值的平方和
        norm_squared = np.sum(diff ** 2)
        # 返回窗函数的值
        return math.exp(-norm_squared / (2 * h ** 2))

    def parzen(self, sample, trainSet):
        # 计算样本在训练集中的似然度
        likelihood = sum(self.window(sample, row, 1) for _, row in trainSet.iterrows())
        # 返回似然度的平均值
        return likelihood / len(trainSet)

    def classify(self, sample):
        # 计算样本属于类别1、2、3的后验概率
        posterior_1 = self.parzen(sample, self.trainSet_1)
        posterior_2 = self.parzen(sample, self.trainSet_2)
        posterior_3 = self.parzen(sample, self.trainSet_3)

        # 打印样本的值和后验概率
        print("样本:", sample.values)
        print("p(w1):", posterior_1)
        print("p(w2):", posterior_2)
        print("p(w3):", posterior_3)

        # 根据后验概率判断样本属于哪个类别
        if posterior_1 > posterior_2:
            if posterior_1 > posterior_3:
                print("样本属于类别1")
            else:
                print("样本属于类别3")
        else:
            if posterior_2 > posterior_3:
                print("样本属于类别2")
            else:
                print("样本属于类别3")
        print("-------------------------------------")

# 使用示例
classifier = ParzenClassifier('exp2_3.xlsx')
classifier.classify(pd.Series([0.5, 1.0, 0.0]))
classifier.classify(pd.Series([0.31, 1.51, -0.50]))
classifier.classify(pd.Series([-0.3, 0.44, -0.1]))

############################################################################################

#k邻近概率密度估计
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS']

from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = pd.read_excel('exp2_3.xlsx', header=[0, 1])

# 一维 k-近邻概率密度估计
def kNN_density_1d(data, feature, k_values):
    x = data[('类3', feature)].values
    x_test = np.linspace(np.min(x), np.max(x), 1000)
    for k in k_values:
        densities = []
        for test_point in x_test:
            distances = np.abs(x - test_point)
            k_nearest_distances = np.sort(distances)[:k]
            h_k = np.max(k_nearest_distances)
            if h_k == 0:  # 处理分母为零的情况
                densities.append(0)
            else:
                V_R = 2 * h_k
                densities.append(k / len(x) / V_R)
        plt.figure()  # 创建新的图形
        plt.plot(x_test, densities)
        plt.xlabel(feature)
        plt.ylabel('概率密度')
        plt.title(f'类别 3 的概率密度估计 - {feature} - k={k}')
        plt.savefig('2_3_1d_'+str(k)+'.png')
        plt.show()


# 二维 k-近邻概率密度估计
def kNN_density_2d(data, feature1, feature2, k_values):
    x1 = data[('类2', feature1)].values
    x2 = data[('类2', feature2)].values
    x_test1, x_test2 = np.meshgrid(np.linspace(np.min(x1), np.max(x1), 100),
                                   np.linspace(np.min(x2), np.max(x2), 100))
    for k in k_values:
        densities = np.zeros_like(x_test1)
        for i in range(len(x_test1)):
            for j in range(len(x_test1[0])):
                test_point = np.array([x_test1[i][j], x_test2[i][j]])
                distances = np.sqrt(np.sum((np.vstack((x1, x2)).T - test_point) ** 2, axis=1))
                k_nearest_distances = np.sort(distances)[:k]
                h_k = np.max(k_nearest_distances)
                if h_k == 0:  # 处理分母为零的情况
                    densities[i][j] = 0
                else:
                    V_R = np.pi * h_k ** 2
                    densities[i][j] = k / len(x1) / V_R
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_test1, x_test2, densities, cmap='viridis')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel('概率密度')
        ax.set_title(f'类别 2 的概率密度估计 - k={k}')
        plt.savefig('2_3_2d_'+str(k)+'.png')
        plt.show()


# 三维 k-近邻概率密度估计
def kNN_density_3d(testData: np.matrix, k: int):
    # 分别获取类别 1、2、3 的三维特征数据
    class1_data = data['类1'].values
    class2_data = data['类2'].values
    class3_data = data['类3'].values

    for test_point in testData:  # 在这里使用传递给函数的 testData
        # 计算测试点与各类别数据点的距离
        dists_class1 = np.sqrt(np.sum((class1_data - test_point) ** 2, axis=1))
        dists_class2 = np.sqrt(np.sum((class2_data - test_point) ** 2, axis=1))
        dists_class3 = np.sqrt(np.sum((class3_data - test_point) ** 2, axis=1))

        # 对距离数组进行排序，并提取出第k个数据
        dists_class1.sort()
        dists_class2.sort()
        dists_class3.sort()

        V1 = 4 * np.pi * (dists_class1[k - 1] ** 3) / 3
        V2 = 4 * np.pi * (dists_class2[k - 1] ** 3) / 3
        V3 = 4 * np.pi * (dists_class3[k - 1] ** 3) / 3

        posterior = [k / 10 / V1, k / 10 / V2, k / 10 / V3]

        print("类条件概率密度数组:" + str(posterior))
        print('属于类' + str(posterior.index(max(posterior)) + 1))



# 一维 k-NN 概率密度估计，类别 3 特征 x1
k_values = [1, 3, 5]
kNN_density_1d(data, 'x1', k_values)

# 二维 k-NN 概率密度估计，类别 2 特征 x1 和 x2
kNN_density_2d(data, 'x1', 'x2', k_values)

test_points1 = [(-0.41, 0.82, 0.88)]
test_points2 = [(0.14, 0.72, 4.1)]
test_points3 = [(-0.81, 0.61, -0.38)]
# 进行三维 k-近邻概率密度估计和类别判断
kNN_density_3d(test_points1,3)
kNN_density_3d(test_points2,3)
kNN_density_3d(test_points3,3)
