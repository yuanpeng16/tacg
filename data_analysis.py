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


def get_map(words):
    id_map = {}
    for group in words:
        for i, word in enumerate(group):
            id_map[word] = i
    return id_map


def convert_to_id(id_map, line):
    return tuple([id_map.get(word, 0) for word in line])


class WordMapper(object):
    def __init__(self):
        action_words = [
            ['look', 'run', 'walk', 'jump'],
            ['left', 'right']
        ]
        words = [
            ['twice', 'thrice'],
            ['and', 'after'],
            ['opposite', 'around'],
            ['turn']
        ]
        self.action_id_map = get_map(action_words)
        self.function_id_map = get_map(words)

    def get_action_ids(self, line):
        return convert_to_id(self.action_id_map, line)

    def get_function_id(self, line):
        return convert_to_id(self.function_id_map, line)


class WordSyntaxChecker(object):
    def __init__(self):
        unequal_syntax_list = [
            ['look', 'turn'],  # look left, turn left
            ['left', 'twice'],  # look left, look twice
            ['left', 'thrice']  # look left, look thrice
        ]
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
                if a == 'look':
                    replace_word = 'walk'
                elif a == 'left':
                    replace_word = 'right'
                elif a == 'right':
                    replace_word = 'left'
                else:
                    continue
                replaced_x = self.get_replaced_output(replace_word, i, x)
                replaced_y = self.get_replaced_output(replace_word, i, y)
                if replaced_x != replaced_y:
                    return False
        return True


class MultipleEqualChecker(object):
    def __init__(self, data_map):
        self.data_map = data_map

    def check(self, x, y):
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

        :param x:
        :param y:
        :return:
        """
        assert len(x) == len(y)
        x = " ".join(x)
        y = " ".join(y)
        input_a = "turn left and look left"
        input_b = "turn opposite left and look"

        first = x == input_a and y == input_b
        second = x == input_b and y == input_a
        if not (first or second):
            return True

        input_c = "look and look"
        input_d = "look opposite left"
        input_c = tuple(input_c.split(" "))
        input_d = tuple(input_d.split(" "))
        if (input_c not in self.data_map) or (input_d not in self.data_map):
            return True
        return False


class WrapperChecker(object):
    def __init__(self, data_map):
        self.checkers = [
            WordSyntaxChecker(),
            ReplacementChecker(data_map),
            MultipleEqualChecker(data_map)
        ]

    def check(self, x, y):
        for checker in self.checkers:
            if not checker.check(x, y):
                return False
        return True


class Checker(object):
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


def analyze(data):
    lines, outputs = data
    word_mapper = WordMapper()
    input_set = {}
    for line, output in zip(lines, outputs):
        reference_syntax = word_mapper.get_function_id(line)
        reference_semantics = word_mapper.get_action_ids(line)
        name = tuple([reference_syntax, reference_semantics, tuple(output)])
        if name not in input_set:
            input_set[name] = []
        input_set[name].append(line)
    data_map = {tuple(line): output for line, output in zip(lines, outputs)}
    checker = Checker(data_map)
    for key, value in input_set.items():
        if len(value) > 1:
            checker.check_pairs(key[:-1], value)
    print(len(lines), len(input_set))


def main():
    fn = 'SCAN/add_prim_split/tasks_train_addprim_jump.txt'
    data = read_data(fn)
    analyze(data)


if __name__ == '__main__':
    main()
