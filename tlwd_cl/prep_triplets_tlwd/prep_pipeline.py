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



if __name__ == '__main__':
    file_root_path = f"../../data/math23k/"
    '将NxCx转换为N，进行交换律正则'
    # file_path = f"{file_root_path}train.jsonl"
    # save_path = f"{file_root_path}train_VU.jsonl"
    # prefix_postfix_norm_NoOn(file_path, save_path)
    'TreeDis with different alpha'
    # preprocess_treedis_tlwd(file_root_path + 'train_cl.jsonl', file_root_path + 'tree_dis_tlwd_alpha0_25.jsonl',
    #                    alpha=0.25)
    # preprocess_treedis_ted(file_root_path + 'train_cl.jsonl', file_root_path + 'tree_dis_ted.jsonl',)
    '先做一个train_cl，然后固定positive'
    preprocess_cl_RecursiveTreeDis(file_root_path,
                                   'train_cl.jsonl',
                                   'tree_dis_ted.jsonl',
                                   41,
                                   'train_cl_ted_seed41.jsonl')
    preprocess_cl_RecursiveTreeDis(file_root_path,
                                   'train_cl.jsonl',
                                   'tree_dis_ted.jsonl',
                                   42,
                                   'train_cl_ted_seed42.jsonl')
    preprocess_cl_RecursiveTreeDis(file_root_path,
                                   'train_cl.jsonl',
                                   'tree_dis_ted.jsonl',
                                   1,
                                   'train_cl_ted_seed1.jsonl')
    #
    # preprocess_treedis(file_root_path + 'train_16191_normed.jsonl', file_root_path + 'tree_dis_New_Normed_Alpha1_5.jsonl',
    #                    alpha=1.5)
    # '先做一个train_cl，然后固定positive'
    # preprocess_cl_RecursiveTreeDis(file_root_path,
    #                                'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha0_25.jsonl',
    #                                'tree_dis_New_Normed_Alpha1_5.jsonl',
    #                                'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1_5.jsonl')
    'check'
    # data1 = load_data(
    #     file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha0_25.jsonl')
    # data2 = load_data(
    #     file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1.jsonl')
    # data3 = load_data(
    #     file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1_5.jsonl')
    # for i in range(100):
    #     negative1 = find_by_id(data1, data1[i]['negative'])
    #     negative2 = find_by_id(data2, data2[i]['negative'])
    #     negative3 = find_by_id(data3, data3[i]['negative'])
    #     print(data1[i]['prefix_normed'], negative1['prefix_normed'], negative2['prefix_normed'],
    #           negative3['prefix_normed'])


    # for i in range(5):
    #     file_root_path = f"../data/cv_asdiv-a_RecursiveTreeDis/fold{i}/"
    #     '将NxCx转换为N，进行交换律正则'
    #     file_path = f"../data/cv_asdiv-a_RecursiveTreeDis/fold{i}/train.jsonl"
    #     save_path = f"../data/cv_asdiv-a_RecursiveTreeDis/fold{i}/train_normed.jsonl"
    #     prefix_postfix_norm(file_path, save_path)
    #     'TreeDis with different alpha'
    #     preprocess_treedis(file_root_path + 'train_normed.jsonl', file_root_path + 'tree_dis_New_Normed_Alpha1_5.jsonl', alpha=1.5)
    #     '先做一个train_cl，然后固定positive'
    #     preprocess_cl_RecursiveTreeDis(file_root_path,
    #                                    'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1.jsonl',
    #                                    'tree_dis_New_Normed_Alpha1_5.jsonl',
    #                                    'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1_5.jsonl')
    #     'check'
    #     data1 = load_data(
    #         file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha0_25.jsonl')
    #     data2 = load_data(
    #         file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1.jsonl')
    #     data3 = load_data(
    #         file_root_path + 'train_cl_NewTreeDis_PosSample10Text_NegRandom10Exp_NoFilter_Alpha1_5.jsonl')
    #     for i in range(100):
    #         negative1 = find_by_id(data1, data1[i]['negative'])
    #         negative2 = find_by_id(data2, data2[i]['negative'])
    #         negative3 = find_by_id(data3, data3[i]['negative'])
    #         print(data1[i]['prefix_normed'], negative1['prefix_normed'], negative2['prefix_normed'], negative3['prefix_normed'])