import os
from group_equals import group_equivalent_pairs


def read_data(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    lines = list(set(lines))
    lines.sort()
    inputs = []
    outputs = []
    for line in lines:
        terms = line.split('OUT:')
        line = terms[0].strip().split(' ')[1:]
        output = terms[1].strip().split(' ')
        inputs.append(line)
        outputs.append(output)
    return [inputs, outputs]


class WordSyntaxChecker(object):
    def __init__(self):
        # look left, turn left
        action_words = ['look', 'run', 'walk']
        direction_words = ['left', 'right']
        unequal_syntax_list = []
        for action in action_words:
            unequal_syntax_list.append([action, 'turn'])
        for direction in direction_words:
            unequal_syntax_list.append([direction, 'twice'])
            unequal_syntax_list.append([direction, 'thrice'])
        unequal_syntax_list.append(['twice', 'thrice'])
        unequal_syntax_list.append(['and', 'after'])
        unequal_syntax_list.append(['opposite', 'around'])
        reverse = [[b, a] for [a, b] in unequal_syntax_list]
        unequal_syntax_list = unequal_syntax_list + reverse
        unequal_syntax_list = [tuple(x) for x in unequal_syntax_list]
        self.unequal_syntax_set = set(unequal_syntax_list)

    def check(self, x, y):
        assert len(x) == len(y)
        for a, b in zip(x, y):
            if (a, b) in self.unequal_syntax_set:
                return False
        return True


def get_key_word_sets(x, y):
    key_pairs = set()
    for a, b in zip(x, y):
        if a == b:
            continue
        key = [a, b]
        key.sort()
        key_pairs.add(tuple(key))
    groups = group_equivalent_pairs(list(key_pairs))
    return groups


class SentenceSyntaxPairChecker(object):
    def __init__(self, data):
        self.data = data
        self.searched_sets = {}

    def search_length(self, key_word_sets):
        key_word_map = {}
        for key, values in enumerate(key_word_sets):
            for value in values:
                key_word_map[value] = key

        inputs, output = self.data
        sentences = {}
        for line, actions in zip(inputs, output):
            length = len(actions)
            key = [key_word_map.get(word, word) for word in line]
            key = tuple(key)
            if key not in sentences:
                sentences[key] = length
            elif sentences[key] != length:
                return False
        return True

    def check(self, x, y):
        key_word_sets = get_key_word_sets(x, y)
        groups = [tuple(sorted(list(group))) for group in key_word_sets]
        groups = tuple(sorted(groups))
        if groups in self.searched_sets:
            return self.searched_sets[groups]
        result = self.search_length(key_word_sets)
        if result:
            self.searched_sets[groups] = result
        return result


class ReferenceManager(object):
    def __init__(self):
        word_lists = [['look', 'walk', 'run', 'jump'], ['left', 'right']]
        word_map = {}
        for i, words in enumerate(word_lists):
            for word in words:
                word_map[word] = i
        self.word_map = word_map

    def get_reference_word(self, word):
        return self.word_map.get(word, word)

    def get_reference_sentence(self, sentence):
        return [self.get_reference_word(word) for word in sentence]


class CheckerCollection(object):
    def __init__(self, data):
        self.checkers = [
            WordSyntaxChecker(),
            SentenceSyntaxPairChecker(data),
        ]
        self.counter = [0] * (len(self.checkers) + 1)

    def check(self, x, y):
        self.counter[0] += 1
        for i, checker in enumerate(self.checkers):
            if not checker.check(x, y):
                return False
            self.counter[i + 1] += 1
        return True

    def get_counter(self):
        return self.counter


class Analyzer(object):
    def __init__(self, data):
        self.data = data
        self.reference_manager = ReferenceManager()
        self.pair_checker = CheckerCollection(data)

    def pass_pair(self, line1, line2, output1, output2):
        assert len(line1) == len(line2)
        assert len(output1) == len(output2)

        r1 = self.reference_manager.get_reference_sentence(line1)
        r2 = self.reference_manager.get_reference_sentence(line2)
        if tuple(r1) == tuple(r2):
            return True

        if not self.pair_checker.check(line1, line2):
            return True
        return False

    def analyze_group(self, lines, outputs):
        n = len(lines)
        for i in range(n):
            for j in range(i + 1, n):
                if not self.pass_pair(lines[i], lines[j], outputs[i],
                                      outputs[j]):
                    return False
        return True

    def analyze(self):
        lines, outputs = self.data
        line_group = {}
        output_group = {}
        for line, output in zip(lines, outputs):
            key = tuple([len(line), len(output)])
            if key not in line_group:
                line_group[key] = []
                output_group[key] = []
            line_group[key].append(line)
            output_group[key].append(output)
        keys = line_group.keys()
        keys = sorted(keys)
        for i, key in enumerate(keys):
            print(i, '/', len(keys), key)
            grouped_lines = line_group[key]
            grouped_outputs = output_group[key]
            if not self.analyze_group(grouped_lines, grouped_outputs):
                return False
        print('checkers:', self.pair_checker.get_counter())
        return True


def main():
    fn = os.path.join('SCAN', 'add_prim_split', 'tasks_train_addprim_jump.txt')
    data = read_data(fn)
    analyzer = Analyzer(data)
    if analyzer.analyze():
        print("All pairs pass the condition.")


if __name__ == '__main__':
    main()
