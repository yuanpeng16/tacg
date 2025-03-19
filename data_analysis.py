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
    id_map = {}
    for group in action_words:
        for word in group:
            id_map[word] = -1
    for group in words:
        for i, word in enumerate(group):
            id_map[word] = i
    return id_map


def check_replacement(key, value, data):
    print(key)
    for x in value:
        print(x)


def analyze(data):
    id_map = get_id_map()
    lines, outputs = data
    input_set = {}
    for line, output in zip(lines, outputs):
        id = tuple([id_map[word] for word in line] + [output])
        if id not in input_set:
            input_set[id] = []
        input_set[id].append(line)
    for key, value in input_set.items():
        if len(value) > 1:
            check_replacement(key[:-1], value, data)
    print(len(lines), len(input_set))


def main():
    fn = 'SCAN/add_prim_split/tasks_train_addprim_jump.txt'
    data = read_data(fn)
    analyze(data)


if __name__ == '__main__':
    main()
