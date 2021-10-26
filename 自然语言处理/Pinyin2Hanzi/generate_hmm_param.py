from ChineseTone import *

PINYIN2HANZI = '../data/pinyin2hanzi.txt'
SENTENCE_FILE = './result/setence.txt'
WORD_FILE = '../data/word.txt'

PY2HZ_FILE = './result/hmm_py2hz.model'
START_FILE = './result/hmm_start.model'
EMISSION_FILE = './result/hmm_emission.model'
TRANSITION_FILE = './result/hmm_transition.model'


PINYIN_NUM = 400.    # 从pinyin2hanzi.txt统计出来的拼音总数
HANZI_NUM  = 6664.   # 从pinyin2hanzi.txt统计出来的汉字总数


def read_from_py2hz(emission):
    print('read from pinyin2hanzi.txt')
    data = {}
    with open(PINYIN2HANZI) as f:
        for line in f.readlines():
            # 空格分隔 去除前后空白
            pinyin, hanzi_str = line.split()
            pinyin = pinyin.strip()
            hanzi_str = hanzi_str.strip()

            # for emission
            for hanzi in hanzi_str:
                emission.setdefault(hanzi, {})
                emission[hanzi][pinyin] = emission[hanzi].get(pinyin, 0) + 1

            # 将pinyin2hanzi保存为字典，方便使用
            if len(pinyin) > 0 and len(hanzi_str) > 0:
                data[pinyin] = hanzi_str

    save_model(data, PY2HZ_FILE)

def read_from_sentence_txt(start, emission, transition):
    print('read from sentence.txt')
    with open(SENTENCE_FILE) as f:
        for line in f.readlines():
            line = line.strip()
            # for start
            if len(line) < 1:
                continue
            first_word = line[0]
            start[first_word] = start.get(first_word, 0) + 1

            # for emission
            pinyin_list = PinyinHelper.convertToPinyinFromSentence(line, pinyinFormat=PinyinFormat.WITHOUT_TONE)
            word_list = [w for w in line]

            for hanzi, pinyin in zip(word_list, pinyin_list):
                emission.setdefault(hanzi, {})
                emission[hanzi][pinyin] = emission[hanzi].get(pinyin, 0) + 1

            # for transition
            for f, t in zip(line[:-1], line[1:]):
                transition.setdefault(f, {})
                transition[f][t] = transition[f].get(t, 0) + 1


def read_from_word_txt(start, emission, transition):
    ## ! 基于word.txt的优化
    print('read from word.txt')
    _base = 1000.
    _min_value = 2.
    for line in open(WORD_FILE):
        line = line.strip()
        if '=' not in line:
            continue
        if len(line) < 3:
            continue
        ls = line.split('=')
        if len(ls) != 2:
            continue
        word, num = ls
        word = word.strip()
        num = num.strip()
        if len(num) == 0:
            continue
        num = float(num)
        num = max(_min_value, num / _base)

        # if not util.is_chinese(word):
        #     continue

        ## for start
        start.setdefault(word[0], 0)
        start[word[0]] += num

        ## for emission
        pinyin_list = PinyinHelper.convertToPinyinFromSentence(word, pinyinFormat=PinyinFormat.WITHOUT_TONE)
        char_list = [c for c in word]
        for hanzi, pinyin in zip(char_list, pinyin_list):
            emission.setdefault(hanzi, {})
            emission[hanzi].setdefault(pinyin, 0)
            emission[hanzi][pinyin] += num

        ## for transition
        for f, t in zip(word[:-1], word[1:]):
            transition.setdefault(f, {})
            transition[f].setdefault(t, 0)
            transition[f][t] += num


def gen_start(start):
    """
    将start字典的频数转换为概率
    {'你'：8，'他'：2}  --> {'你'：0.8，'他'：0.2}
    """
    data = {'default': 1, 'data': None}
    count = HANZI_NUM
    for hanzi in start:
        count += start[hanzi]
    for hanzi in start:
        start[hanzi] = start[hanzi] / count

    data['default'] = 1.0 / count
    data['data'] = start
    save_model(data, START_FILE)


def gen_emission(emission):
    """
    将emission字典的频数转换为概率
    {'了'：{'liao':1,'le':1},'你':{'li':1}}  --> {'了'：{'liao':0.5,'le':0.5},'你':{'li':1.0}}
    """
    data = {'default': 1.e-200, 'data': None}
    for hanzi in emission:
        num_sum = 0.
        for pinyin in emission[hanzi]:
            num_sum += emission[hanzi][pinyin]
        for pinyin in emission[hanzi]:
            emission[hanzi][pinyin] = emission[hanzi][pinyin] / num_sum

    data['data'] = emission
    save_model(data, EMISSION_FILE)


def gen_transition(transition):
    """
    将transition字典的频数转换为概率
    {'京'：{'都'：10，'东'：40}}  --> {'京'：{'都'：0.2，'东'：0.8}}
    """
    data = {'default': 1./HANZI_NUM, 'data': None}
    for w1 in transition:
        num_sum = HANZI_NUM  # 默认每个字都有可能
        for w2 in transition[w1]:
            num_sum += transition[w1][w2]

        for w2 in transition[w1]:
            transition[w1][w2] = float(transition[w1][w2] + 1) / num_sum  # 加一平滑
        transition[w1]['default'] = 1. / num_sum

    data['data'] = transition
    save_model(data, TRANSITION_FILE)


def save_model(word_dict, model_path):
    print("开始将字典写入", model_path)
    f = open(model_path, 'w')
    f.write(str(word_dict))
    f.close()


def count(words):
    with open(SENTENCE_FILE) as f:
        for line in f.readlines():
            line = line.strip()
            for word in line:
                words.add(word)


if __name__ == '__main__':
    start = {}          # {'你'：1310，'他'：123}
    emission = {}       # {'了'：{'liao':1,'le':1},'你':{'li':1}}
    transition = {}     # {'京'：{'都'：12，'东'：32}}

    read_from_py2hz(emission)
    read_from_sentence_txt(start, emission, transition)
    read_from_word_txt(start, emission, transition)

    gen_start(start)
    gen_emission(emission)
    gen_transition(transition)