import argparse
from prep_tree_dis import *
from prep_cl import *
# from ..utils.utils import *

def load_json(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data, filename):
    f = open(filename, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()


def prefix_postfix_norm(file_path, save_path):
    data = load_json(file_path)
    ops = ['+', '-', '*', '/', '^']
    for d in data:
        prefix_op = [op if op in ops else 'N' for op in d['prefix']]
        postfix_op = [op if op in ops else 'N' for op in d['postfix']]
        # d['prefix_treedis'] = prefix_treedis
        # d['postfix_treedis'] = postfix_treedis

        prefix_tree = build_tree(prefix_op)
        subtree_norm(prefix_tree)
        prefix_normed = preorder_traversal(prefix_tree)
        d['prefix_normed'] =prefix_normed

        d['postfix_normed'] = from_prefix_to_postfix(prefix_normed)
        print(d['prefix'], d['prefix_normed'], d['postfix_normed'])

    save_json(data, save_path)


def prefix_postfix_norm_NoOn(file_path, save_path):
    data = load_json(file_path)
    ops = ['+', '-', '*', '/', '^']
    for d in data:
        # prefix_op = [op if op in ops else 'N' for op in d['prefix']]
        # postfix_op = [op if op in ops else 'N' for op in d['postfix']]
        d['prefix_normed'] = [op if op in ops else 'N' for op in d['prefix']]
        d['postfix_normed'] = [op if op in ops else 'N' for op in d['postfix']]
        # print(d['prefix'], d['prefix_normed'], d['postfix_normed'])

    save_json(data, save_path)


def find_by_id(data, id):
    ans = None
    for d in data:
        if d['id'] == id:
            ans = d
            break
    return ans


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--similarity', default='TLWD', type=str)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    file_root_path = f"/home/wanglu/mwps_code_wrap/data/mathqa/"
    'TreeDis with different alpha'
    # preprocess_treedis_tlwd(file_root_path + 'train_cl.jsonl', file_root_path + 'tree_dis_tlwd_alpha0_25.jsonl',
    #                    alpha=0.25)
    # preprocess_treedis_ted(file_root_path + 'train_cl.jsonl', file_root_path + 'tree_dis_ted.jsonl',)
    '先做一个train_cl，然后固定positive'
    if args.similarity == 'TLWD':
        preprocess_cl_RecursiveTreeDis(file_root_path,
                                   'train.jsonl',
                                   'tree_dis_tlwd_alpha0_25.jsonl',
                                   args.seed,
                                   f'train_cl_tlwd_alpha0_25_seed{args.seed}.jsonl')
    elif args.similarity == 'TED':
        preprocess_cl_TED(file_root_path,
                          'train.jsonl',
                          'tree_dis_ted.jsonl',
                          args.seed,
                          f'train_cl_ted_seed{args.seed}.jsonl')