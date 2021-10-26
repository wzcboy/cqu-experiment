dataFilePath = './data/train.txt'
dict1_filePath = './model/dict1.model'
dict2_filePath = './model/dict2.model'

dict1 = {}
dict2 = {}


n = 2

import math

def train(dataFilePath):
    k = 0
    with open(dataFilePath) as f:
        for sentence in f.readlines():
            words_list = sentence.strip().split()
            words_list.insert(0, "<BOS>")
            words_list.append("<EOS>")
            # 去除空的项
            words_list = list(filter(None, words_list))

            create_ngrams(words_list, n - 1, dict1)
            create_ngrams(words_list, n, dict2)

            k = k + 1
            print("执行到第", k, "行")



def create_ngrams(words_list, n, dict_ngrams):
    """
    形成n元组字典
    :param words_list: 一个句子形成的词表，如["江泽民"，"发表"，"新年讲话"]
    :param n:
    :param dict_ngrams: n元组字典
    :return:
    """
    for i in range(len(words_list) - n + 1):
        ngrams = ""
        for j in range(i, i + n):
            if j == (i + n - 1):
                ngrams += " "
            ngrams += words_list[j]
        ngrams = ngrams.strip()
        dict_ngrams[ngrams] = dict_ngrams.get(ngrams, 0) + 1

def save_model(word_dict, model_path):
    f = open(model_path, 'w')
    f.write(str(word_dict))
    f.close()


if __name__ == '__main__':
    print("[Info]start training......")
    train(dataFilePath)
    save_model(dict1, dict1_filePath)
    save_model(dict2, dict2_filePath)
    print("[Info]end training!")