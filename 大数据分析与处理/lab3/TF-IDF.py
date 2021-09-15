import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import linear_model, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# 导入网上购物数据
data = pd.read_csv('online_shopping_10_cats.csv')
print("原始数据集大小为：", data.shape)


def mypreprocess(data):
    """
    对数据进行预处理
    :param data: DataFrame类型的数据
    :return: none
    """
    # 预处理
    for i in data.index:
        cat = data.loc[i, "cat"]
        label = data.loc[i, "label"]
        review = data.loc[i, "review"]
        if pd.isna(cat) or pd.isna(label) or pd.isna(review):
            data.drop([i], inplace=True)

    # 重置index
    data.reset_index(drop=True, inplace=True)
    print("预处理后数据集大小为：", data.shape)


def getStopWords():
    """
    获取停用词列表，这里使用哈工大停用词表
    :return: list of stopwords
    """
    # 导入停用词表
    stopWordsFile = open('hit_stopwords.txt')

    # 经过统计，哈工大停用词表大小为1561个字符
    stopwords = []
    for line in stopWordsFile.readlines():
        stopword = [c for c in line.strip()]
        stopwords.extend(stopword)

    # 根据具体文本，再添加停用词
    stopwords.extend([str(i) for i in range(20)])
    stopwords.extend(['!', '...', '2008', ' ', '都', '这家', '我们', 'in', '这个', '什么', '星'])

    return stopwords


def getTFIDF(df_data, stopwords):
    """
    获取文本-词语矩阵，矩阵值为tf-idf值
    :param df_data: DataFrame类型的数据
    :param stopwords: 停用词列表
    :return: TF-IDF矩阵
    """
    # 构造字符串列表的语料库
    # 比如"我 爱 北京 天安门"
    corpus = []
    for i in df_data.index:
        words = [w for w in jieba.cut(df_data.loc[i, "review"]) if w not in stopwords]
        # 用空格分开
        str_words = " ".join(words)
        corpus.append(str_words)

    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer(max_features=10000)

    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)

    # 获取词袋中所有文本关键词
    allWord = vectorizer.get_feature_names()
    print("所有关键词个数为：", len(allWord))
    print(allWord)

    # 将词频矩阵X统计成TF-IDF值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)

    return tfidf


if __name__ == '__main__':
    mypreprocess(data)
    stopwords = getStopWords()

    tfidf_data = getTFIDF(data, stopwords)
    tfidf_label = list(data['label'])

    x_train, x_test, y_train, y_test = train_test_split(tfidf_data, tfidf_label, test_size=0.4, random_state=42)

    # Logisitic Regression
    lr = linear_model.LogisticRegression(
        class_weight='balanced',
        solver='newton-cg',
        fit_intercept=True
    ).fit(x_train, y_train)

    # 预测结果，并给出f1_score
    y_pred = lr.predict(x_test)
    print("\nLogisticRegression分类结果：")
    print("Precision", precision_score(y_test, y_pred, average='binary'))
    print("Recall", recall_score(y_test, y_pred, average='binary'))
    print("F1_score", f1_score(y_test, y_pred, average='binary'))

    # 构造SGD线性分类器
    sgd_huber = linear_model.SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        alpha=20,
        loss='modified_huber',
        class_weight='balanced'
    ).fit(x_train, y_train)

    # 预测结果，并给出f1_score
    y_pred = sgd_huber.predict(x_test)
    print("\nSGD分类结果：")
    print("Precision", precision_score(y_test, y_pred, average='binary'))
    print("Recall", recall_score(y_test, y_pred, average='binary'))
    print("F1_score", f1_score(y_test, y_pred, average='binary'))

    # 构造SVM分类器
    svm_classifier = svm.SVC(gamma='auto')
    svm_classifier.fit(x_train, y_train)

    y_pred = svm_classifier.predict(x_test)
    print("\nSVM分类结果：")
    print("Precision", precision_score(y_test, y_pred, average='binary'))
    print("Recall", recall_score(y_test, y_pred, average='binary'))
    print("F1_score", f1_score(y_test, y_pred, average='binary'))