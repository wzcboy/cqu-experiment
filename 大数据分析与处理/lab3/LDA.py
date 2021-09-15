import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
import jieba
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn import svm

# 导入网上购物数据
data = pd.read_csv('online_shopping_10_cats.csv')
print("原始数据集大小为：", data.shape)

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

# 导入停用词表
stopWordsFile = open('hit_stopwords.txt')

# 经过统计，哈工大停用词表大小为1561个字符
stopwords = []
for line in stopWordsFile.readlines():
    stopword = [c for c in line.strip()]
    stopwords.extend(stopword)

# 关闭文件流
stopWordsFile.close()

# 根据具体文本，再添加停用词
stopwords.extend([str(i) for i in range(20)])
stopwords.extend(['!', '...', '2008', ' ', '都', '这家', '我们', 'in', '这个', '什么', '星'])


# 将所有文档进行分词，装入二维数组
words_list = []
for i in data.index:
    words = [w for w in jieba.cut(data.loc[i, "review"]) if w not in stopwords]
    words_list.append(words)

# 构造词典
dictionary = corpora.Dictionary(words_list)

# 去掉过大或者过小的词
dictionary.filter_extremes(no_below=10, no_above=0.35)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in words_list]

# lda模型，num_topics设置主题的个数 passes循环次数，次数越高，越准确
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=15, passes=10)

# 保存模型
lda.save('lda_train.model')

# lda = models.LdaModel.load('lda_train.model')

# 打印所有主题，每个主题显示5个词
for topic in lda.print_topics(num_words=5):
    print(topic)

# 获取文档-主题向量
doc_topic_vec = []

for i in range(len(data)):
    top_topics = lda.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(15)]
    topic_vec.extend([len(data.iloc[i].review)])
    doc_topic_vec.append(topic_vec)

# 将文档-主题向量添加到data中
data['topic'] = doc_topic_vec

# 训练一个有监督的分类器

# 划分训练集和测试集，比例为6：4
df_train, df_test = train_test_split(data, test_size=0.4, random_state=42)

# 重置index
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

x_train = np.array(list(df_train['topic']))
y_train = np.array(list(df_train['label']))

x_test = np.array(list(df_test['topic']))
y_test = np.array(list(df_test['label']))

# Scale Data：归一化
scaler = preprocessing.StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)


# Logisitic Regression
lr = linear_model.LogisticRegression(
    class_weight='balanced',
    solver='newton-cg',
    fit_intercept=True
).fit(x_train_scale, y_train)

# 预测结果，并给出f1_score
y_pred = lr.predict(x_test_scale)
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
).fit(x_train_scale, y_train)

# 预测结果，并给出f1_score
y_pred = sgd_huber.predict(x_test_scale)
print("\nSGD分类结果：")
print("Precision", precision_score(y_test, y_pred, average='binary'))
print("Recall", recall_score(y_test, y_pred, average='binary'))
print("F1_score", f1_score(y_test, y_pred, average='binary'))

# 构造SVM分类器
svm_classifier = svm.SVC(gamma='auto')
svm_classifier.fit(x_train_scale, y_train)

# 预测结果，并给出f1_score
y_pred = svm_classifier.predict(x_test_scale)
print("\nSVM分类结果：")
print("Precision", precision_score(y_test, y_pred, average='binary'))
print("Recall", recall_score(y_test, y_pred, average='binary'))
print("F1_score", f1_score(y_test, y_pred, average='binary'))