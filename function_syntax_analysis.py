import os
from sentence_syntax_analysis import read_data
from sentence_syntax_analysis import SentenceSyntaxPairChecker


def get_key_list():
    keys_list = ['twice', 'thrice', 'and', 'before', 'around', 'opposite', 'turn']
    key_list = []
    for i, a in enumerate(keys_list):
        for b in keys_list[i + 1:]:
            key_list.append([a, b])
    return key_list


def main():
    fn = os.path.join('SCAN', 'add_prim_split', 'tasks_train_addprim_jump.txt')
    data = read_data(fn)
    checker = SentenceSyntaxPairChecker(data)

    key_list = get_key_list()
    for x, y in key_list:
        if not checker.check(x, y):
            print(x, y, "doe not pass the condition.")
    print("All function word pairs pass the condition.")


if __name__ == '__main__':
    main()
