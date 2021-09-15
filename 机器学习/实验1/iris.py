import numpy as np
from sklearn.datasets import load_iris

# 载入数据集
iris = load_iris()
X = np.concatenate((iris.data.T, [[1 for i in range(150)]]), axis=0)  # 获取花卉两列数据集
Y = iris.target

# 划分训练集和测试集
X_train = np.concatenate((X[:, :40], X[:, 50:90], X[:, 100:140]), axis=1)
X_test = np.concatenate((X[:, 40:50], X[:, 90:100], X[:, 140:]), axis=1)
Y_train = np.concatenate((Y[:40], Y[50:90], Y[100:140]), axis=0)
Y_test = np.concatenate((Y[40:50], Y[90:100], Y[140:]), axis=0)

def NewtonMethod(X, Y, beta):
    """
    牛顿法迭代计算对数几率回归
    :param X: 输入的训练数据
    :param Y: 输入的标签结果
    :param beta: 初始参数值
    :return: 返回beta和迭代次数
    """
    L = 0
    n = 0
    length = len(Y)
    while 1:
        beta_T_x = np.dot(beta.T[0], X)   # beta.T 是对其进行矩阵转置 1*len的矩阵
        new_L = 0
        for i in range(length):
            new_L = new_L + (-Y[i]*beta_T_x[i] + np.log(1 + np.exp(beta_T_x[i])))  # P59 式3.27

        #迭代终止条件
        if np.abs(L - new_L) <= 0.00001:
            break

        # 牛顿法迭代更新beta
        n = n + 1
        L = new_L

        dbeta = 0     # 一阶导数
        d2beta = 0    # 二阶导数
        for i in range(length):
            dbeta = dbeta - np.dot(np.array([X[:, i]]).T, (Y[i] - (np.exp(beta_T_x[i]) / (1+np.exp(beta_T_x[i])))))
            d2beta = d2beta + np.dot(np.array([X[:, i]]).T, np.array([X[:, i]]).T.T) * \
                     (np.exp(beta_T_x[i])/(1+np.exp(beta_T_x[i]))) * (1-(np.exp(beta_T_x[i])/(1+np.exp(beta_T_x[i]))))
        beta = beta - np.dot(np.linalg.inv(d2beta), dbeta)
    return beta, n

# setosa(0) 和 versicolor(1) 形成的f1分类器
X1 = X_train[:, :80]
Y1 = Y_train[:80]
beta1 = np.array([[0], [0], [0], [0], [1]])
beta1, n1 = NewtonMethod(X1, Y1, beta1)
print("分类器1的模型参数是：", beta1)
print("迭代次数：", n1)
print()

# setosa(0) 和 virginica(2) 形成的f2分类器
X2 = np.concatenate((X_train[:, :40], X_train[:, 80:]), axis=1)
Y2 = np.concatenate((Y_train[:40], Y_train[80:]), axis=0)
Y2[40:] -= 1   # 转化为 0 1
beta2 = np.array([[0], [0], [0], [0], [1]])
beta2, n2 = NewtonMethod(X2, Y2, beta2)
print("分类器2的模型参数是：", beta2)
print("迭代次数：", n2)
print()

# versicolor(1) 和 virginica(2) 形成的f3分类器
X3 = X_train[:, 40:]
Y3 = Y_train[40:]
Y3 -= 1        # 转化为 0 1
beta3 = np.array([[0], [0], [0], [0], [1]])
beta3, n3 = NewtonMethod(X3, Y3, beta3)
print("分类器3的模型参数是：", beta3)
print("迭代次数：", n3)
print()

# 结果预测
def unit_step_func(X):
    res = []
    for i in X:
        if i > 0:
            res.append(1)
        elif i == 0:
            res.append(0.5)
        else:
            res.append(0)
    return np.array(res)


# f1分类器预测结果
beta1_T_x = np.dot(beta1.T[0], X_test)
f1_predict = unit_step_func(beta1_T_x)
# print(f1_predict)

# f2分类器预测结果
beta2_T_x = np.dot(beta2.T[0], X_test)
f2_predict = unit_step_func(beta2_T_x)
for i in range(len(f2_predict)):
    if f2_predict[i] == 1:
        f2_predict[i] = 2
# print(f2_predict)

# f3分类器预测结果
beta3_T_x = np.dot(beta3.T[0], X_test)
f3_predict = unit_step_func(beta3_T_x)
f3_predict = f3_predict + 1
# print(f3_predict)

# 三个分类器投票
all_predict = []
for i in range(len(f1_predict)):
    count = [0, 0, 0]
    count[f1_predict[i]] += 1
    count[f2_predict[i]] += 1
    count[f3_predict[i]] += 1
    maxPredict = count.index(max(count))
    all_predict.append(maxPredict)

print("测试集的实际结果：", list(Y_test))
print("测试集的预测结果：", all_predict)

# 性能评估

# 计算精度
cnt = 0
for i in range(len(Y_test)):
    if Y_test[i] == all_predict[i]:
        cnt += 1
accuracy = np.round(cnt / len(Y_test), 4)

print("accuracy: {}%".format(accuracy*100))
