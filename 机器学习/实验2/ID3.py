"""
使用CART方法实现决策树
包括连续值和离散值
"""
import copy
import plotJsonTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
trainData = pd.read_excel("decisionTree_train_data.xlsx", header=0, encoding="utf-8")
trainData.set_index('编号', inplace=True)    # 去除默认index
trainData = trainData.reset_index(drop=True)
trainData.index += 1

testData = pd.read_excel("decisionTree_predict_data.xlsx", header=0, encoding="utf-8")
testData.set_index('编号', inplace=True)    # 去除默认index
testData = testData.reset_index(drop=True)
testData.index += 1

# 处理数据
def get_full_labels(data):
    """
    获取所以属性及其所有取值
    :param data: 数据集
    :return: 字典
    """
    full_labels = {}
    for feature in data.columns[:-1]:
        full_labels[feature] = list(data[feature].unique())
    return full_labels
# 所有属性及其所有属性取值
full_labels = get_full_labels(trainData)

def Gain(data, a, target):
    """
    信息增益计算
    :param data: input the name of DataFrame
    :param a: the name of feature
    :param target: the name of target
    :return: gain值
    """
    gain = 0
    length = len(data)
    labels = data[target].value_counts()
    for label in data[target].unique():
        p = labels[label] / length
        gain -= p * np.log2(p)

    attributes = data[a].value_counts()
    for name in data[a].unique():
        print(name)
        kindDict = {}  # 对于每个属性值， 记录每个种类的对应数目
        for kind in data[target].unique():  # initial
            kindDict[kind] = 0

        for index in data.index:
            if data[a].values[index - 1] == name:
                kindDict[data[target].values[index - 1]] += 1

        ent = 0
        for key in kindDict.keys():
            p = kindDict[key] / attributes[name]
            if p != 0:
                ent -= p * np.log2(p)

        gain -= (attributes[name] / length) * ent
    return gain


def chooseBestFeature(data):
    """
    从所有属性中选择最优划分属性
    :param data: 数据集输入
    :return: 最优划分属性和划分点（连续值的时候）
    """
    gain_max = 0
    pos = 0
    for feature in data.columns[:-1]:
        print('属性', feature)
        gain = Gain(data, feature, "好瓜")

        print("信息增益为", gain)
        print()
        if gain > gain_max:
            gain_max = gain
            pos_best = pos
            feature_best = feature

    print("最大信息增益为：", gain_max, "属性名称：", feature_best)
    print("所以应该基于{}划分".format(feature_best))
    return feature_best, pos_best

def transLabel(label):
    if label == '是':
        return '好瓜'
    else:
        return '坏瓜'


def JudgeAllEqual(data):
    """
    判断data中是否所以样本的属性值都相等
    :param data: 数据集
    :return: True or False
    """
    flag = 1
    labels = list(data.columns[:-1])
    firstline = list(data[label].values[0] for label in labels)
    for index in data.index:
        line = list(data[label].values[index-1] for label in labels)
        if line != firstline:
            flag = 0
            break
    return flag

def splitSubData(data, bestFeatureName, a):
    """
    按照属性值来划分子集
    :param data: 数据集
    :param bestFeatureName: 属性名称
    :param a: 属性的取值
    :return: 划分后的数据集
    """
    Dv = data[0:0]
    for index in data.index:
        if data[bestFeatureName].values[index - 1] == a:
            Dv = Dv.append(data[index - 1:index])

    Dv = Dv.drop(columns=[bestFeatureName])
    Dv = Dv.reset_index(drop=True)
    Dv.index += 1
    return Dv

def classify(decisionTree, oneTestData):
    """
    给一条测试集打标签
    :param decisionTree: 决策树（Json格式）
    :param testData: DataFrame格式的一条数据
    :return: 类别
    """
    # 当递归到字符串的时候，说明是标签值，直接返回
    if type(decisionTree) == str:
        return decisionTree
    featureName = list(decisionTree.keys())[0]
    value = oneTestData[featureName].values[0]
    subTree = decisionTree[featureName][value]
    return classify(subTree, oneTestData)

def calAccuracy(decisionTree, testData):
    """
    计算模型精度
    :param decisionTree: 生成的决策树（Json格式）
    :param testData: 测试集
    :return: 精度
    """
    rightCount = 0
    for index in testData.index:
        oneTestData = testData[index-1:index]
        predictLabel = classify(decisionTree, oneTestData)
        if predictLabel == oneTestData["好瓜"].values[0]:
            rightCount += 1
    return rightCount / len(testData)

def countRightNum(data, label):
    """
    统计data中标签为label的个数
    :param data: 数据集
    :param label: 标签值
    :return: 个数
    """
    count = 0
    for index in data.index:
        if data["好瓜"].values[index-1] == label:
            count += 1
    return count

def TreeGenerate(data):
    """
    不带剪枝的决策树生成
    :param data: 数据集
    :return: 决策树（Json格式）
    """
    # 当data中样本全属于同一类别
    if len(data["好瓜"].unique()) == 1:
        print(data["好瓜"].values[0], "\n")
        return data["好瓜"].values[0]
    # 属性集合为空，或者data中样本在features中取值全部相等
    if len(data.columns) == 1 or JudgeAllEqual(data):
        print(data["好瓜"].value_counts(sort=True).index[0], "\n")
        return data["好瓜"].value_counts(sort=True).index[0]

    bestFeatureName, at = chooseBestFeature(data)
    print("最优特征为：", bestFeatureName)


    Tree = {bestFeatureName: {}}
    for a in full_labels[bestFeatureName]:
        dataCopy = copy.deepcopy(data)
        Dv = splitSubData(dataCopy, bestFeatureName, a)

        print("\n", bestFeatureName, a, "\n", Dv)

        if len(Dv) == 0:
            mainLabel = data["好瓜"].value_counts(sort=True).index[0]
            print("******", bestFeatureName, a, mainLabel)
            Tree[bestFeatureName][a] = mainLabel
        else:
            print("******", bestFeatureName, a, "?")
            Tree[bestFeatureName][a] = TreeGenerate(Dv)
    return Tree

def TreeGenerate_prepruning(trainData, testData):
    """
    采用预剪枝的决策树生成
    :param trainData: 训练集
    :param testData: 测试集
    :return: 决策树（Json格式）
    """
    # 当data中样本全属于同一类别
    if len(trainData["好瓜"].unique()) == 1:
        print(trainData["好瓜"].values[0], "\n")
        return trainData["好瓜"].values[0]
    # 样本数目最多的那个分类
    mainLabel = trainData["好瓜"].value_counts(sort=True).index[0]
    # 属性集合为空，或者data中样本在features中取值全部相等
    if len(trainData.columns) == 1 or JudgeAllEqual(trainData):
        print(mainLabel, "\n")
        return mainLabel

    bestFeatureName, at = chooseBestFeature(trainData)
    print("最优特征为：", bestFeatureName)


    # 计算划分前的正确预测的数目
    baseRightNum = countRightNum(testData, mainLabel)
    # 划分后正确预测的数目
    splitRightNum = 0

    for a in full_labels[bestFeatureName]:
        Dv = splitSubData(trainData, bestFeatureName, a)
        Dv_prune = splitSubData(testData, bestFeatureName, a)
        if len(Dv) == 0:
            splitRightNum += countRightNum(Dv_prune, mainLabel)
        else:
            splitRightNum += countRightNum(Dv_prune, Dv["好瓜"].value_counts(sort=True).index[0])

    # 继续划分
    if baseRightNum < splitRightNum:
        Tree = {bestFeatureName: {}}
    # 停止划分, 即剪枝
    else:
        return mainLabel

    for a in full_labels[bestFeatureName]:
        Dv = splitSubData(trainData, bestFeatureName, a)
        Dv_prune = splitSubData(testData, bestFeatureName, a)
        print("\n", bestFeatureName, a, "\n", Dv)

        if len(Dv) == 0:
            print("******", bestFeatureName, a, mainLabel)
            Tree[bestFeatureName][a] = mainLabel
        else:
            print("******", bestFeatureName, a, "?")
            Tree[bestFeatureName][a] = TreeGenerate_prepruning(Dv, Dv_prune)
    return Tree

def TreeGenerate_postpruning(trainData, testData):
    """
    采用后剪枝的决策树生成
    :param trainData: 训练集
    :param testData: 测试集
    :return: 决策树（Json格式）
    """
    # 当data中样本全属于同一类别
    if len(trainData["好瓜"].unique()) == 1:
        print(trainData["好瓜"].values[0], "\n")
        return trainData["好瓜"].values[0]
    # 样本数目最多的那个分类
    mainLabel = trainData["好瓜"].value_counts(sort=True).index[0]
    # 属性集合为空，或者data中样本在features中取值全部相等
    if len(trainData.columns) == 1 or JudgeAllEqual(trainData):
        print(mainLabel, "\n")
        return mainLabel

    bestFeatureName, at = chooseBestFeature(trainData)
    print("最优特征为：", bestFeatureName)


    Tree = {bestFeatureName: {}}

    for a in full_labels[bestFeatureName]:
        Dv = splitSubData(trainData, bestFeatureName, a)
        Dv_prune = splitSubData(testData, bestFeatureName, a)
        print("\n", bestFeatureName, a, "\n", Dv)

        if len(Dv) == 0:
            print("******", bestFeatureName, a, mainLabel)
            Tree[bestFeatureName][a] = mainLabel
        else:
            print("******", bestFeatureName, a, "?")
            Tree[bestFeatureName][a] = TreeGenerate_postpruning(Dv, Dv_prune)


    # 后剪枝部分
    # 计算划分前的正确预测的数目
    baseRightNum = countRightNum(testData, mainLabel)
    # 划分后正确预测的数目
    splitRightNum = 0

    for a in full_labels[bestFeatureName]:
        Dv = splitSubData(trainData, bestFeatureName, a)
        Dv_prune = splitSubData(testData, bestFeatureName, a)
        if len(Dv) == 0:
            splitRightNum += countRightNum(Dv_prune, mainLabel)
        else:
            splitRightNum += countRightNum(Dv_prune, Dv["好瓜"].value_counts(sort=True).index[0])

    # 停止划分，即剪枝
    if baseRightNum > splitRightNum:
        return mainLabel

    return Tree

# 生成Json格式的决策树
features = list(trainData.columns[0:-1])
decisionTree1 = TreeGenerate(trainData)
decisionTree2 = TreeGenerate_prepruning(trainData, testData)
decisionTree3 = TreeGenerate_postpruning(trainData, testData)
print("未剪枝产生的树：", decisionTree1)
print("预剪枝产生的树：", decisionTree2)
print("后剪枝产生的树：", decisionTree3)

print("未剪枝的验证集精度：", calAccuracy(decisionTree1, testData))
print("预剪枝的验证集精度：", calAccuracy(decisionTree2, testData))
print("后剪枝的验证集精度：", calAccuracy(decisionTree3, testData))


# 绘制决策树
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plotJsonTree.createPlot(decisionTree1)
plotJsonTree.createPlot(decisionTree2)
plotJsonTree.createPlot(decisionTree3)