class DefaultHmmParams():
    def __init__(self):
        self.py2hz_dict = self.load_model('./result/hmm_py2hz.model')
        self.start_dict = self.load_model('./result/hmm_start.model')
        self.emission_dict = self.load_model('./result/hmm_emission.model')
        self.transition_dict = self.load_model('./result/hmm_transition.model')

    # 加载预训练模型
    def load_model(self, model_path):
        f = open(model_path, 'r')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    def start(self, state):
        default = self.start_dict['default']
        data = self.start_dict['data']

        if state in data:
            prob = data[state]
        else:
            prob = default
        return float(prob)

    def emission(self, state, observation):
        pinyin = observation
        hanzi = state

        default = self.emission_dict['default']
        data = self.emission_dict['data']

        if hanzi not in data:
            return float(default)

        prob_dict = data[hanzi]

        if pinyin in prob_dict:
            return float(prob_dict[pinyin])
        else:
            return float(default)

    def transition(self, from_state, to_state):
        default = self.transition_dict['default']
        data = self.transition_dict['data']

        if from_state not in data:
            return float(default)

        prob_dict = data[from_state]

        if to_state in prob_dict:
            return float(prob_dict[to_state])
        elif 'default' in prob_dict:
            return float(prob_dict['default'])
        else:
            return float(default)


    def get_states(self, observation):
        """得到拼音对应的所有可能汉字"""
        return [hanzi for hanzi in self.py2hz_dict[observation]]