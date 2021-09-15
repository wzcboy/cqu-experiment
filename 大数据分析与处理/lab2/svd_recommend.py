import numpy as np
import matplotlib.pyplot as plt
import math

# macos中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

trainData = np.load('训练集矩阵.npy')
valData = np.load('验证集矩阵.npy')
testData = np.load('测试集矩阵.npy')

userList = np.load('用户列表.npy')
movieList = np.load('电影列表.npy')

print("训练集大小为", trainData.shape)
print("验证集大小为", valData.shape)
print("测试集大小为", testData.shape)


def myfit(A, n_factor=10, n_epoch=10, learning_rate=0.005):
    # 选取的factor
    K = n_factor

    m, n = A.shape
    # P = np.random.randint(1, 5, [m, K])
    # Q = np.random.randint(1, 5, [K, n])

    P = np.random.randn(m, K)
    Q = np.random.randn(K, n)

    # 迭代次数
    epoch = n_epoch
    # 学习率
    lr = learning_rate
    rmse_list = []
    for e in range(epoch):
        sse = []
        for u in range(m):
            for i in range(n):
                if A[u][i] == 0:
                    continue
                r_ui = np.dot(P[u], Q[:, i])
                e_ui = A[u][i] - r_ui
                sse.append(e_ui * e_ui)
                # update
                P[u] = P[u] + lr * e_ui * Q[:, i]
                Q[:, i] = Q[:, i] + lr * e_ui * P[u]

        rmse = math.sqrt(sum(sse)/len(sse))
        rmse_list.append(rmse)
        print("第" + str(e) + "次迭代RMSE为：", rmse)

    # 绘制RMSE变化折线图
    x = np.arange(1, len(rmse_list)+1)
    plt.plot(x, rmse_list)
    plt.title("K={}, 学习率={}".format(K, lr))
    plt.xlabel("迭代次数")
    plt.ylabel("RMSE")
    plt.show()

    return P, Q


def predict(u, i, P, Q):
    est = np.dot(P[u], Q[:, i])
    # 控制在[0,5]
    if est < 1:
        return 1
    elif est > 5:
        return 5
    else:
        return est


def test(testData, P, Q):
    m, n = testData.shape
    sse = []
    predictions = []
    for u in range(m):
        for i in range(n):
            if testData[u][i] == 0:
                continue
            pre_r = predict(u, i, P, Q)
            e = testData[u][i] - pre_r
            sse.append(e * e)
            predictions.append([u, i, testData[u][i], pre_r])

    rmse = math.sqrt(sum(sse) / len(sse))
    print("误差为", rmse)

    return predictions


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = {}
    for uid, _, true_r, est in predictions:
        if uid in user_est_true:
            user_est_true[uid].append((est, true_r))
        else:
            user_est_true[uid] = [(est, true_r)]

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


if __name__ == '__main__':
    # 学习率不变，改变K值来观察
    # for i in range(20, 110, 10):
    #     print("------------------------------")
    #     print("K取值为：", i)
    #     print("开始训练......")
    #     P, Q = myfit(trainData, n_factor=i)
    #     print("开始验证......")
    #     predictions = test(valData, P, Q)
    #     precisions, recalls = precision_recall_at_k(predictions)
    #     print("------------------------------")

    # K值不变，改变学习率来观察
    # learning_rate_list = [0.02, 0.025, 0.03]
    # learning_rate_list = [0.002, 0.005, 0.01, 0.015]
    # for i in learning_rate_list:
    #     print("------------------------------")
    #     print("K取值为：", 10)
    #     print("学习率取值为：", i)
    #     print("开始训练......")
    #     P, Q = myfit(trainData, learning_rate=i)
    #     print("开始验证......")
    #     predictions = test(valData, P, Q)
    #     precisions, recalls = precision_recall_at_k(predictions)
    #     print("------------------------------")

    print("------------------------------")
    print("K取值为：", 10)
    print("学习率取值为：", 0.01)
    print("开始训练......")
    P, Q = myfit(trainData, n_factor=10, learning_rate=0.01)
    print("开始验证......")
    predictions = test(testData, P, Q)
    precisions, recalls = precision_recall_at_k(predictions)
    print("------------------------------")

# prediction = np.dot(P, Q)
# prediction_flatten = prediction[trainData.nonzero()]
# train_data_flatten = trainData[trainData.nonzero()]
# SSE = math.sqrt(mean_squared_error(prediction_flatten, train_data_flatten))

