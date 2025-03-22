def read_data(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    lines = list(set(lines))
    inputs = []
    outputs = []
    for line in lines:
        terms = line.split('OUT:')
        line = terms[0].strip().split(' ')[1:]
        output = terms[1].strip()
        inputs.append(line)
        outputs.append(output)
    return [inputs, outputs]


class WordMapper(object):
    def __init__(self, swap_list):
        words = [
            ['twice', 'thrice'],
            ['and', 'after'],
            ['opposite', 'around'],
            ['turn']
        ]
        id_map = {}
        for swap, group in zip(swap_list, words):
            for i, word in enumerate(group):
                id_map[word] = 1 - i if swap else i
        self.function_id_map = id_map

    def get_function_id(self, line):
        return tuple([self.function_id_map.get(word, 0) for word in line])


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


class ReplacementChecker(object):
    def __init__(self, data_map):
        self.data_map = data_map
        action_words = ['look', 'run', 'walk']
        direction_words = ['left', 'right']
        replace_words = {}
        for i, x in enumerate(action_words):
            replace_words[x] = action_words[i - 1]
        for i, x in enumerate(direction_words):
            replace_words[x] = direction_words[i - 1]
        self.replace_words = replace_words

    def get_replaced_output(self, replace_word, position, x):
        replace_input = [w for w in x]
        replace_input[position] = replace_word
        replace_input = tuple(replace_input)
        assert replace_input in self.data_map
        return self.data_map[replace_input]

    def check(self, x, y):
        assert len(x) == len(y)
        for i, [a, b] in enumerate(zip(x, y)):
            if a == b:
                if a in self.replace_words:
                    replace_word = self.replace_words[a]
                else:
                    continue
                replaced_x = self.get_replaced_output(replace_word, i, x)
                replaced_y = self.get_replaced_output(replace_word, i, y)
                if replaced_x != replaced_y:
                    return False
        return True


class MultipleEqualChecker(object):
    """We consider the following pair.
        - turn left and look left
        - turn opposite left and look
    For them to have the same syntax representation, it requires syntax that

    left = opposite, and = left, look = and, left = look

    Therefore, left = opposite = and = look (1)
    On the other hand, we consider the following pair.
        - look and look
        - look opposite left
    If (1) holds, they have the same syntax. However, they have different output lengths. The upper one is two and the lower one is three. So, (1) does not hold, and the syntax of the original pair is different.
    """

    def __init__(self, data_map):
        self.data_map = data_map
        action_words = ['look', 'run', 'walk']
        direction_words = ['left', 'right']
        pairs = []
        replace_inputs = []
        for action in action_words:
            for direction in direction_words:
                pairs.append([
                    ['turn', direction, 'and', action, direction],
                    ['turn', 'opposite', direction, 'and', action]
                ])
                replace_inputs.extend([
                    [action, 'and', action],
                    [action, 'opposite', direction]
                ])
        self.pair_map = set()
        for x, y in pairs:
            x = tuple(x)
            y = tuple(y)
            self.pair_map.add(tuple([x, y]))
            self.pair_map.add(tuple([y, x]))

        for x in replace_inputs:
            assert tuple(x) in self.data_map

    def check(self, x, y):
        assert len(x) == len(y)
        key = tuple([tuple(x), tuple(y)])
        if key not in self.pair_map:
            return True
        return False


class WrapperChecker(object):
    def __init__(self, data_map):
        self.checkers = [
            WordSyntaxChecker(),
            ReplacementChecker(data_map),
            MultipleEqualChecker(data_map)
        ]
        self.counter = [0] * len(self.checkers)

    def check(self, x, y):
        for i, checker in enumerate(self.checkers):
            if not checker.check(x, y):
                return False
            self.counter[i] += 1
        return True

    def get_counter(self):
        return self.counter


class Checker(object):
    """Show that two training samples have different syntax representations,
     given all training predictions are correct.
    """

    def __init__(self, data_map):
        self.checker = WrapperChecker(data_map)

    def check_pairs(self, key, value):
        equal_pairs = []
        for i, x in enumerate(value):
            for y in value[i + 1:]:
                if self.checker.check(x, y):
                    equal_pairs.append([x, y])
        if len(equal_pairs) > 0:
            print(key)
            for a, b in equal_pairs:
                print(a)
                print(b)
                print()

    def get_count(self):
        return self.checker.get_counter()


def analyze(data, swap_list):
    lines, outputs = data
    word_mapper = WordMapper(swap_list)
    input_set = {}
    for line, output in zip(lines, outputs):
        reference_syntax = word_mapper.get_function_id(line)
        name = tuple([reference_syntax, tuple(output)])
        if name not in input_set:
            input_set[name] = []
        input_set[name].append(line)
    data_map = {tuple(line): output for line, output in zip(lines, outputs)}
    checker = Checker(data_map)
    for key, value in input_set.items():
        if len(value) > 1:
            checker.check_pairs(key[:-1], value)
    print(len(lines), len(input_set))
    print(checker.get_count())


def main():
    fn = 'SCAN/add_prim_split/tasks_train_addprim_jump.txt'
    data = read_data(fn)
    swap_list = [0, 0, 0, 0]
    analyze(data, swap_list)


if __name__ == '__main__':
    main()
