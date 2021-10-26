import math
from Pinyin2Hanzi.hmm_implement import DefaultHmmParams
from Pinyin2Hanzi.get_test import *


def viterbi(hmmparams, observations, min_prob=3.14e-200):
    assert(isinstance(hmmparams, DefaultHmmParams))
    V = [{}]
    path = [{}]
    t = 0
    cur_obs = observations[t]

    # 步1 Initialize(t=0)
    cur_states = hmmparams.get_states(cur_obs)
    for state in cur_states:
        score = max(hmm_params.start(state), min_prob) * max(hmm_params.emission(state, cur_obs), min_prob)
        V[0][state] = score
        path[0][state] = 0

    # 步 2 run Viterbi for t > 0
    for t in range(1, len(observations)):
        cur_obs = observations[t]

        V.append({})
        path.append({})

        prev_states = cur_states
        cur_states = hmm_params.get_states(cur_obs)

        for cur in cur_states:
            max_score = 0

            for prev in prev_states:
                score = V[t-1][prev] * max(hmm_params.transition(prev, cur), min_prob) * \
                                       max(hmm_params.emission(cur, cur_obs), min_prob)
                if score > max_score:
                    max_score = score
                    max_prev = prev
            V[t][cur] = max_score
            path[t][cur] = max_prev

    # 步3 Viterbi end
    final_word = max(V[t], key=V[t].get)

    # 步4 path back
    result = []
    prev_word = final_word
    for i in range(1, len(observations)+1):
        t = len(observations) - i
        result.append(prev_word)
        prev_word = path[t][prev_word]

    result.reverse()
    return result


def test(hmm_params):
    # print("-------------------------开始测试------------------------")
    print("---------转化结果---------")
    pinyin_list, hanzi_list = get_test_data()
    for observation in pinyin_list:
        observation = observation.lower().split()
        result = viterbi(hmm_params, observation)
        result = ''.join(result)
        print(result)



if __name__ == '__main__':
    hmm_params = DefaultHmmParams()
    test(hmm_params)