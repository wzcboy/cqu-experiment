"""
统计拼音2汉字文件中
汉字的总数：6664
拼音的总数：400
"""
PY2HZ_FILE = '../data/pinyin2hanzi.txt'
WORD_FILE = '../data/word.txt'

states = set()         # 汉字
observations = set()   # 拼音
with open(PY2HZ_FILE) as f:
    for line in f.readlines():
        pinyin, hanzi_list = line.split()
        print(pinyin)
        observations.add(pinyin)
        for hanzi in hanzi_list:
            states.add(hanzi)

print(len(states))
print(len(observations))

with open(WORD_FILE) as f:
    for line in f.readlines():
        line = line.strip()
        if '=' not in line:
            continue
        if len(line) < 3:
            continue
        word, num = line.split('=')
        word = word.strip()
        num = num.strip()
        for hanzi in word:
            states.add(hanzi)

print(len(states))
print(len(observations))