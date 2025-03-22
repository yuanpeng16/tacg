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


def get_id_map():
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
    action_id_map = get_map(action_words)
    function_id_map = get_map(words)
    return action_id_map, function_id_map


unequal_syntax_list = [
    ['look', 'turn'],  # look left, turn left
    ['left', 'twice'],  # look left, look twice
    ['left', 'thrice']  # look left, look thrice
]
reverse = [[b, a] for [a, b] in unequal_syntax_list]
unequal_syntax_list = unequal_syntax_list + reverse
unequal_syntax_list = [tuple(x) for x in unequal_syntax_list]
unequal_syntax_set = set(unequal_syntax_list)


def check_equal_syntax(x, y):
    assert len(x) == len(y)
    for a, b in zip(x, y):
        if (a, b) in unequal_syntax_set:
            return False
    return True


def get_replaced_output(replace_word, position, x, data_map):
    replace_input = [w for w in x]
    replace_input[position] = replace_word
    replace_input = tuple(replace_input)
    assert replace_input in data_map
    return data_map[replace_input]


def check_replacing_samples(x, y, data_map):
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
            replaced_x = get_replaced_output(replace_word, i, x, data_map)
            replaced_y = get_replaced_output(replace_word, i, y, data_map)
            if replaced_x != replaced_y:
                return False
    return True


def check_multiple_equal(x, y, data_map):
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
    :param data_map:
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
    if (input_c not in data_map) or (input_d not in data_map):
        return True
    return False


def check_binary(key, value, data_map):
    equal_pairs = []
    for i, x in enumerate(value):
        for y in value[i + 1:]:
            if check_equal_syntax(x, y):
                equal_pairs.append([x, y])

    current = equal_pairs
    equal_pairs = []
    for x, y in current:
        if check_replacing_samples(x, y, data_map) and check_multiple_equal(x, y, data_map):
            equal_pairs.append([x, y])

    if len(equal_pairs) > 0:
        print(key)
        for a, b in equal_pairs:
            print(a)
            print(b)
            print()


def check_replacement(key, value, data_map):
    check_binary(key, value, data_map)


def contain_action_words(action_id_map, line):
    for word in line:
        if word in action_id_map:
            return True
    return False


def convert_to_id(id_map, line):
    return tuple([id_map.get(word, 0) for word in line])


def analyze(data):
    action_id_map, function_id_map = get_id_map()
    lines, outputs = data
    input_set = {}
    for line, output in zip(lines, outputs):
        if not contain_action_words(action_id_map, line):
            continue
        reference_syntax = convert_to_id(function_id_map, line)
        reference_semantics = convert_to_id(action_id_map, line)
        id = tuple([reference_syntax, reference_semantics, tuple(output)])
        if id not in input_set:
            input_set[id] = []
        input_set[id].append(line)
    data_map = {tuple(line): output for line, output in zip(lines, outputs)}
    for key, value in input_set.items():
        if len(value) > 1:
            check_replacement(key[:-1], value, data_map)
    print(len(lines), len(input_set))


def main():
    fn = 'SCAN/add_prim_split/tasks_train_addprim_jump.txt'
    data = read_data(fn)
    analyze(data)


if __name__ == '__main__':
    main()
