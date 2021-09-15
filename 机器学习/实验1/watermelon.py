import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 载入数据集
excelPath = '/Users/mac/Downloads/the_project_xiguw.xls'
data = pd.read_excel(excelPath, 'Sheet1')

X = np.array([list(data[u'密度']), list(data[u'含糖量']), [1 for i in range(17)]])   # 3 * 17的矩阵
Y = np.array(data[u"好瓜"])

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y, test_size=0.25, random_state=42)
X_train = X_train.T
X_test = X_test.T
# 初始参数值
beta = np.array([[0], [0], [1]])     # (w, b)


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
        beta_T_x = np.dot(beta.T[0], X)   # beta.T 是对其进行矩阵转置 1*17的矩阵
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


beta, n = NewtonMethod(X_train, Y_train, beta)
print("模型参数是：", beta)
print("迭代次数：", n)

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

# 分类器预测结果
beta_T_x = np.dot(beta.T[0], X_test)
predict = unit_step_func(beta_T_x)

print()
print("测试集的实际结果：", Y_test)
print("测试集的预测结果：", predict)

# 性能评估
# 计算精度
cnt = 0
TP = 0  # 真正例
FP = 0  # 假正例
FN = 0  # 假反例
for i in range(len(Y_test)):
    if Y_test[i] == predict[i]:
        cnt += 1
    if Y_test[i] == 1 and predict[i] == 1 :
        TP += 1
    elif Y_test[i] == 1 and predict[i] == 0 :
        FN += 1
    elif Y_test[i] == 0 and predict[i] == 1 :
        FP += 1

accuracy = np.round(cnt / len(Y_test), 4)
precison = np.round(TP/(TP + FP), 4)
recall = np.round(TP/(TP + FN), 4)
print("accuracy: {}%".format(accuracy*100))
print("precison: {}%".format(precison*100))
print("recall: {}%".format(recall*100))



