"""
针对鸢尾花数据集的BP神经网络训练
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
# 获取花卉数据集
iris = load_iris()
X = iris.data[:, :]
label = iris.target

# 将标签转化为向量
Y = []
label2Vec = {'0': [1, 0, 0], '1': [0, 1, 0], '2': [0, 0, 1]}
for i in range(len(label)):
    Y.append(label2Vec[str(label[i])])
Y = np.array(Y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

learning_rate = 0.2   # 学习率

m, n = x_train.shape
def BP_standard(input_d, input_q, input_l):
    # ------- 在（0，1）范围内随机初始化网络中所有连接权和阈值 --------
    d = input_d  # 输入层神经元个数
    q = input_q  # 隐藏层神经元个数
    l = input_l  # 输出层神经元个数

    gamma = [random.random() for i in range(q)]  # 隐藏层的阈值
    theta = [random.random() for i in range(l)]  # 输出层的阈值

    v = [[random.random() for i in range(q)] for j in range(d)]  # 输入层到隐藏层的连接权矩阵 d * q
    w = [[random.random() for i in range(l)] for j in range(q)]  # 隐藏层到输出层的连接权矩阵 q * l
    print("初始参数值为：")
    print("gamma:", gamma)
    print("theta:", theta)
    print("v:", v)
    print("w:", w)
    # ------- 循环迭代 训练BP网络中的参数 --------
    count = 0  # 迭代次数
    lastE = 0
    losslist = []
    while True:
        count += 1
        # if count % 1000 == 0:
        # print(count, lastE)
        elist = []
        for i in range(m):
            # 根据当前参数和式（5.3）计算当前样本的输出yk
            alpha = np.dot(x_train[i], v)     # np.dot(1 * d, d * q) = 1 * q
            b = sigmoid(alpha - gamma, 1)  # 1 * q
            beta = np.dot(b, w)         # np.dot(1 * q, q * l) = 1 * l
            yk = sigmoid(beta - theta, 1)  # 预测的y值

            # 在该样本上的均分误差
            E = sum((yk - y_train[i]) * (yk - y_train[i])) / 2
            elist.append(E)
            # 根据式（5.10）计算输出层神经元的梯度项gj
            g = yk * (1 - yk) * (y_train[i] - yk)     # 1 * l
            e = b * (1 - b) * np.dot(w, g.T).T  # 1 * q

            # 根据式（5.11）- （5.14）更新连接权
            w += learning_rate * np.dot(b.reshape(q, 1), g.reshape(1, l))     # np.dot(q * 1, 1 * l) = q * l
            theta += -learning_rate * g
            v += learning_rate * np.dot(x_train[i].reshape(d, 1), e.reshape(1, q))  # np.dot(d * 1, 1 * q) = d * q
            gamma += -learning_rate * e
        losslist.append(np.mean(elist))
        if np.abs(E - lastE) < 0.0000001 and count > 10:
            break
        lastE = E
    ##Loss可视化
    plt.figure()
    plt.plot([i + 1 for i in range(count)], losslist)
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w, theta, v, gamma, count


def BP_accumulate(input_d, input_q, input_l):
    # ------- 在（0，1）范围内随机初始化网络中所有连接权和阈值 --------
    d = input_d  # 输入层神经元个数
    q = input_q  # 隐藏层神经元个数
    l = input_l  # 输出层神经元个数

    gamma = [[random.random() for i in range(q)] for j in range(m)]  # 隐藏层的阈值 m * q
    theta = [[random.random() for i in range(l)] for j in range(m)]  # 输出层的阈值 m * l

    v = [[random.random() for i in range(q)] for j in range(d)]  # 输入层到隐藏层的连接权矩阵 d * q
    w = [[random.random() for i in range(l)] for j in range(q)]  # 隐藏层到输出层的连接权矩阵 q * l
    print("初始参数值为：")
    print("gamma:", gamma)
    print("theta:", theta)
    print("v:", v)
    print("w:", w)
    # ------- 循环迭代 训练BP网络中的参数 --------
    count = 0  # 迭代次数
    lastE = 0
    losslist = []
    while True:
        count += 1
        if count % 1000 == 0:
            print(count, lastE)
        # 根据当前参数和式（5.3）计算当前样本的输出yk
        alpha = np.dot(x_train, v)           # np.dot(m * d, d * q) = m * q
        b = sigmoid(alpha - gamma, 2)  # m * q
        beta = np.dot(b, w)            # np.dot(m * q, q * l) = m * l
        yk = sigmoid(beta - theta, 2)  # 预测的y值 m * l

        # 在该样本上的均分误差
        E = np.mean((yk - y_train) * (yk - y_train)) / 2
        losslist.append(E)
        # 根据式（5.10）计算输出层神经元的梯度项gj
        g = yk * (1 - yk) * (y_train - yk)        # m * l
        e = b * (1 - b) * np.dot(w, g.T).T  # m * q

        # 根据式（5.11）- （5.14）更新连接权
        w += learning_rate * np.dot(b.T, g)     # np.dot(q * m, m * l) = q * l
        theta += -learning_rate * g
        v += learning_rate * np.dot(x_train.T, e)     # np.dot(d * m, m * q) = d * q
        gamma += -learning_rate * e

        if np.abs(E - lastE) < 0.000001:
            break
        lastE = E
    ##Loss可视化
    plt.figure()
    plt.plot([i + 1 for i in range(count)], losslist)
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w, theta, v, gamma, count


def sigmoid(x, dimension):
    if dimension == 1:
        for i in range(len(x)):
            x[i] = 1 / (1 + math.exp(-x[i]))
    else:
        for i in range(len(x)):
            x[i] = sigmoid(x[i], dimension - 1)
    return x


# ------------BP训练----------------
print("【Info】开始训练BP神经网络......")
w, theta, v, gamma, count = BP_standard(n, n + 1, 3)
print("训练结束!")
print("迭代次数为：", count, "\n")
# w, theta, v, gamma, count = BP_accumulate(n, n + 1, 3)
# theta = (np.sum(theta, axis=0))/m
# gamma = (np.sum(gamma, axis=0))/m
# print("训练结束!")
# print("迭代次数为：", count, "\n")

# ------------用训练好的模型进行预测--------------
def predict(testX):
    alpha = np.dot(testX, v)    # np.dot(1 * d, d * q) = 1 * q
    b = sigmoid(alpha - gamma, 1)  # 使用已经训练好的隐藏层阈值gamma
    beta = np.dot(b, w)         # np.dot(1 * q, q * l) = 1 * l
    yk = sigmoid(beta - theta, 1)  # 使用已经训练好的输出层阈值theta
    return yk


# 欧式距离
def Edist(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (np.square(a[i] - b[i]))
    return np.sqrt(dist)


testNum = x_test.shape[0]
print("开始测试......")
print("测试样本有", testNum, "个")
rightCount = 0
for i in range(testNum):
    predict_vec = predict(x_test[i])

    # 初始化
    dist = 999999
    result_vec = label2Vec['0']
    # 遍历所有可能的标签，找出距离最小的作为预测的向量
    for key, vec in label2Vec.items():
        if(len(predict_vec) != len(vec)):
            print("Error ", predict_vec, vec)
        temp_dist = Edist(predict_vec, vec)
        if temp_dist < dist:
            dist = temp_dist
            result_vec = vec

    if list(result_vec) == list(y_test[i]):
        rightCount += 1

precison = rightCount / testNum
print("准确率为：", precison)