import numpy as np
from matplotlib import pyplot as plt


def load_dataset(filename):
    dataset = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            # 将特征值提取出来，并转换成浮点数
            features = [float(x) for x in line[:3]]  # 前三列是特征值
            # 将标签提取出来
            label = line[3]  # 最后一列是标签
            dataset.append(features)
            labels.append(label)
    return np.array(dataset), np.array(labels)

# 加载数据
dataset, labels = load_dataset('exp2_4.txt')

# 数据预处理
def normalize_dataset(dataset):
    min_vals = dataset.min(axis=0)
    max_vals = dataset.max(axis=0)
    ranges = max_vals - min_vals
    # 使用公式进行归一化处理
    norm_dataset = (dataset - min_vals) / ranges
    return norm_dataset

# 可视化数据
def visualize_data(dataset, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {'largeDoses': 'r', 'smallDoses': 'g', 'didntLike': 'b'}

    for i in range(len(dataset)):
        ax.scatter(dataset[i, 0], dataset[i, 1], dataset[i, 2], c=colors[labels[i]], marker='o')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.savefig('2_4.png')
    plt.show()

norm_dataset = normalize_dataset(dataset)

visualize_data(norm_dataset, labels)

def split_dataset(dataset, labels, test_ratio=0.1):
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = np.random.choice(len(dataset), test_set_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(dataset)), test_indices)
    train_dataset = dataset[train_indices]
    train_labels = labels[train_indices]
    test_dataset = dataset[test_indices]
    test_labels = labels[test_indices]
    return train_dataset, train_labels, test_dataset, test_labels

train_dataset, train_labels, test_dataset, test_labels = split_dataset(norm_dataset, labels)

def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2) ** 2))
def mahalanobis_distance(instance1, instance2):
    diff = instance1 - instance2
    covariance_matrix = np.cov(instance1, instance2)
    cov_inv = np.linalg.inv(covariance_matrix)
    distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
    return distance

class KNN:
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_instance in X_test:
            distances = []
            for i, train_instance in enumerate(self.X_train):
                if self.distance == 'euclidean':
                    dist = euclidean_distance(test_instance, train_instance)
                elif self.distance == 'mahalanobis':
                    dist = mahalanobis_distance(test_instance, train_instance)
                distances.append((self.y_train[i], dist))
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:self.k]
            labels = [neighbor[0] for neighbor in neighbors]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        return predictions

    def accuracy(self, y_true, y_pred):
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

# 构建并训练模型
knn = KNN(k=3, distance='euclidean')
knn.fit(train_dataset, train_labels)

# 在测试集上进行预测
predictions = knn.predict(test_dataset)

# 计算准确率
accuracy = knn.accuracy(test_labels, predictions)
print(f'Accuracy: {accuracy:.2f}')
