import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
    return X, Y

def load_cifar10(ROOT):
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_cifar10_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def preprocess_data(X):
    X = X / 255.0  # 归一化到 0-1
    return X

X_train, y_train, X_test, y_test = load_cifar10('cifar-10-batches-py')
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 划分验证集
X_val = X_train[-1000:]
y_val = y_train[-1000:]
X_train = X_train[:-1000]
y_train = y_train[:-1000]

# 取测试集 1000 样本
X_test = X_test[:1000]
y_test = y_test[:1000]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def relu_derivative(self, Z):
        return Z > 0

    def forward(self, X):
        X = X.reshape(X.shape[0], -1)  # Reshape the input data to be 2D
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        log_probs = -np.log(A2[range(m), Y])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y):
        m = X.shape[0]

        X_flattened = X.reshape(m, -1)  # 将输入数据展平为 (N, 3072) 的形状

        # 输出层梯度
        dZ2 = self.A2
        dZ2[range(m), Y] -= 1
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X_flattened.T, dZ1) / m  # 使用展平后的输入数据
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 更新参数
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        losses = []
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            epoch_loss = 0
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # 前向传播
                A2 = self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_batch, A2)
                epoch_loss += loss

                # 反向传播
                self.backward(X_batch, y_batch)

            epoch_loss /= (X_train.shape[0] // batch_size)
            losses.append(epoch_loss)

            # 每个 epoch 后在验证集上评估模型
            val_preds = self.predict(X_val)
            val_accuracy = self.compute_accuracy(y_val, val_preds)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 绘制损失图
        plt.plot(range(1, epochs + 1), losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    def test(self, X_test, y_test):
        test_preds = self.predict(X_test)
        test_accuracy = self.compute_accuracy(y_test, test_preds)
        print(f'Test Accuracy: {test_accuracy:.4f}')

# 创建神经网络模型
model = NeuralNetwork(input_size=3072, hidden_size=1024, output_size=10, learning_rate=0.01)

# 训练模型
model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=64)

# 在测试集上评估模型
model.test(X_test, y_test)
