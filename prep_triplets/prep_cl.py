import copy
import json
import numpy as np
import random
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


'''Sim_BiBLEU for Text-based Retrieval Strategy'''


def sim(s1, s2):
    s1 = [x for x in s1]
    s2 = [x for x in s2]
    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu(
        references=[s1],
        hypothesis=s2,
        smoothing_function=smooth
    )
    bleu2 = sentence_bleu(
        references=[s2],
        hypothesis=s1,
        smoothing_function=smooth
    )
    return (bleu1 + bleu2) / 2


'''id, text, original_text, prefix, infix, postfix, nums, answer
    length: 21161 这里丢弃了一部分，可能是论文中采用的QR和RODA只能覆盖这么多样本 => 拿出来了2k分别作为验证集和测试集了
    Math23K一共有23162个样本，且只有id, original_text, segmented_text, equation, ans'''

def preprocess_cl(data_root_path, only_asdiv = False):
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + 'train.jsonl')
    '''(TED)'''
    treedis_data = load_data(data_root_path + 'tree_dis.jsonl')
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''建立全零矩阵，保存equation-equation之间的TED'''
    treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''
    expr_expr_reverse_dict = {x: i for i, x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        '''
        [i, j] in treedis_matrix 分别指向两个expression在expr_id_dict的index
        下方公式是论文中Sim_eq(E1, E2)的计算公式，那说明d[1]存的是TED(E1, E2)
        到此为止treedis_data的结构就知晓了['E1;E2', TED(E1, E2)]
        '''
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)
    '''filter_ids存储所有length<10的expression在expr_id_dict中对应的id'''
    filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
    # treedis_matrix1 = copy.deepcopy(treedis_matrix)
    treedis_matrix2 = copy.deepcopy(treedis_matrix)
    for i in range(len(treedis_matrix)):
        '''negative infinity， 每个expression和其自身的distance应该为无穷小'''
        treedis_matrix2[i][i] = -float('inf')
    # for idx in filter_ids:
    #     treedis_matrix1[:, idx] = -float('inf')
    #     treedis_matrix1[idx, :] = -float('inf')
    for idx in filter_ids:
        '''有点奇怪，所有length<10的expression和其他expression之间的的distaance记为无穷小'''
        treedis_matrix2[:, idx] = -float('inf')
        treedis_matrix2[idx, :] = -float('inf')

    res = []
    if only_asdiv:
        real_data = [d for d in train_data if d['id'] > 1000 and d['id'] < 3000]
        res = [d for d in train_data if d['id'] <= 1000 or d['id'] >= 3000]
    else: real_data = train_data
    original_len = len(res)
    for d in real_data:
        postfix_positive = ' '.join(d['postfix'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        # if len(positive_ids) == 1:
        #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
        #     positive_ids = expr_id_dict[postfix_positive]
        minscore = float('inf')
        positive = d['id']
        for idx in positive_ids:
            '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
            if idx != d['id'] and sim(src, cached[idx]) < minscore:
                minscore = sim(src, cached[idx])
                positive = idx
        '选出和当前expression具有最高TED的expr，值得注意的是由于treedis_matrix2[i][i]==无穷小，所以选出来的实际上是次高值'
        # list[] => max index => 多个最大值则返回第一个
        postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]
        negative_ids = expr_id_dict[postfix_negative]
        maxscore = -float('inf')
        negative = d['id']
        for idx in negative_ids:
            '选出具有最大Sim_BiBLEU的expr'
            if idx != d['id'] and sim(src, cached[idx]) > maxscore:
                maxscore = sim(src, cached[idx])
                negative = idx
        d['positive'] = positive
        d['negative'] = negative
        res.append(d)
        if len(res) % 100 == 0:
            print((len(res) - original_len) / len(train_data))


    f = open(data_root_path + 'train_cl.jsonl', 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()



def preprocess_cl_NewTreeDis(data_root_path, only_asdiv=False):
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + 'train_normed.jsonl')
    '''(TED)'''
    treedis_data = load_data(data_root_path + 'tree_dis_New_Normed_Alpha0_1.jsonl')
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        # d['posifix'] = d['postfix_treedis']
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix_normed']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix_normed'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix_normed'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''建立全零矩阵，保存equation-equation之间的TED'''
    treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''
    expr_expr_reverse_dict = {x: i for i, x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        '''
        [i, j] in treedis_matrix 分别指向两个expression在expr_id_dict的index
        下方公式是论文中Sim_eq(E1, E2)的计算公式，那说明d[1]存的是TED(E1, E2)
        到此为止treedis_data的结构就知晓了['E1;E2', TED(E1, E2)]
        '''
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)
    '''filter_ids存储所有length<10的expression在expr_id_dict中对应的id'''
    for x in expr_id_dict.items():
        length = len(x[1])
    filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
    # treedis_matrix1 = copy.deepcopy(treedis_matrix)
    treedis_matrix2 = copy.deepcopy(treedis_matrix)
    for i in range(len(treedis_matrix)):
        '''negative infinity， 每个expression和其自身的distance应该为无穷小'''
        treedis_matrix2[i][i] = -float('inf')
    # for idx in filter_ids:
    #     treedis_matrix1[:, idx] = -float('inf')
    #     treedis_matrix1[idx, :] = -float('inf')
    for idx in filter_ids:
        '''有点奇怪，所有出现频次<10的expression和其他expression之间的的distaance记为无穷小
            这样做会导致频次<10的expression再treedis_matrix2中被记录为无穷小，导致其永远无法被选为负样本
        '''
        treedis_matrix2[:, idx] = -float('inf')
        treedis_matrix2[idx, :] = -float('inf')

    res = []
    if only_asdiv:
        real_data = [d for d in train_data if d['id'] > 1000 and d['id'] < 3000]
        res = [d for d in train_data if d['id'] <= 1000 or d['id'] >= 3000]
    else:
        real_data = train_data
    original_len = len(res)
    for d in real_data:
        postfix_positive = ' '.join(d['postfix_normed'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        if len(positive_ids) > 10:
            positive_ids = random.sample(positive_ids, 10)
        # if len(positive_ids) == 1:
        #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
        #     positive_ids = expr_id_dict[postfix_positive]
        minscore = float('inf')
        positive = d['id']
        for idx in positive_ids:
            '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
            if idx != d['id'] and sim(src, cached[idx]) < minscore:
                minscore = sim(src, cached[idx])
                positive = idx
        '选出和当前expression具有最高TED的expr，值得注意的是由于treedis_matrix2[i][i]==无穷小，所以选出来的实际上是次高值'
        '从可选neg_postfix_index中随机抽样'
        # random_max_index = random.choice(np.where(treedis_matrix2[exprpos] == np.max(treedis_matrix2[exprpos]))[0])
        # postfix_negative = expr_expr_reverse_dict[random_max_index]

        postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]
        negative_ids = expr_id_dict[postfix_negative]
        if len(negative_ids) > 10:
            negative_ids = random.sample(negative_ids, 10)
        maxscore = -float('inf')
        negative = d['id']
        for idx in negative_ids:
            '选出具有最大Sim_BiBLEU的expr'
            if idx != d['id'] and sim(src, cached[idx]) > maxscore:
                maxscore = sim(src, cached[idx])
                negative = idx
        d['positive'] = positive
        d['negative'] = negative
        res.append(d)
        if len(res) % 10 == 0:
            print((len(res) - original_len) / len(train_data))

    f = open(data_root_path + 'train_cl_NewTreeDis_NormedPostfix_Sample10_RandomNeg_Alpha0_1.jsonl', 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()



def preprocess_cl_RecursiveTreeDis(data_root_path, train_file, treedis_file, seed, save_file):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    print(f"seed is {seed}")
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + train_file)
    '''(TED)'''
    treedis_data = load_data(data_root_path + treedis_file)
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        # d['posifix'] = d['postfix_treedis']
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix_normed']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix_normed'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix_normed'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''建立全零矩阵，保存equation-equation之间的TED'''
    treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''
    expr_expr_reverse_dict = {x: i for i, x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        '''
        [i, j] in treedis_matrix 分别指向两个expression在expr_id_dict的index
        下方公式是论文中Sim_eq(E1, E2)的计算公式，那说明d[1]存的是TED(E1, E2)
        到此为止treedis_data的结构就知晓了['E1;E2', TED(E1, E2)]
        '''
        # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)
        # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] * (1 + np.log2(1 + abs(len1//2 - len2//2)))
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1]

    '''filter_ids存储所有length<10的expression在expr_id_dict中对应的id'''

    # filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
    treedis_matrix2 = copy.deepcopy(treedis_matrix)
    for i in range(len(treedis_matrix)):
        '''negative infinity， 每个expression和其自身的distance应该为无穷小
            新算法中应该改为无穷大'''
        treedis_matrix2[i][i] = -float('inf')
        # treedis_matrix2[i][i] = float('inf')

    # for idx in filter_ids:
    #     '''有点奇怪，所有出现频次<10的expression和其他expression之间的的distaance记为无穷小
    #         这样做会导致频次<10的expression再treedis_matrix2中被记录为无穷小，导致其永远无法被选为负样本
    #     '''
    #     treedis_matrix2[:, idx] = -float('inf')
    #     treedis_matrix2[idx, :] = -float('inf')

    res = []

    for d in train_data:
        postfix_positive = ' '.join(d['postfix_normed'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        if len(positive_ids) > 10:  # 如果positive_ids > 10则随机抽样增加多样性
            positive_ids = random.sample(positive_ids, 10)
        # if len(positive_ids) == 1:
        #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
        #     positive_ids = expr_id_dict[postfix_positive]
        minscore = float('inf')
        positive = d['id']
        for idx in positive_ids:
            '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
            if idx != d['id'] and sim(src, cached[idx]) < minscore:
                minscore = sim(src, cached[idx])
                positive = idx
        d['positive'] = positive
        '选出和当前expression具有最高TED的expr，值得注意的是由于treedis_matrix2[i][i]==无穷小，所以选出来的实际上是次高值'
        '从可选neg_postfix_index中随机抽样'
        # 最接近的expression如果有多个则随机抽样
        random_max_index = random.choice(np.where(treedis_matrix2[exprpos] == np.max(treedis_matrix2[exprpos]))[0])
        postfix_negative = expr_expr_reverse_dict[random_max_index]
        # postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]
        negative_ids = expr_id_dict[postfix_negative]
        # if len(negative_ids) > 10:
        #     negative_ids = random.sample(negative_ids, 10)
        maxscore = -float('inf')
        negative = d['id']
        for idx in negative_ids:
            '选出具有最大Sim_BiBLEU的expr'
            if idx != d['id'] and sim(src, cached[idx]) > maxscore:
                maxscore = sim(src, cached[idx])
                negative = idx
        d['negative'] = negative
        res.append(d)
        if len(res) % 100 == 0:
            print(len(res)/len(train_data))

    f = open(data_root_path + save_file, 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()


def preprocess_cl_TED(data_root_path, train_file, treedis_file, seed, save_file):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    print(f"seed is {seed}")
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + train_file)
    '''(TED)'''
    treedis_data = load_data(data_root_path + treedis_file)
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''建立全零矩阵，保存equation-equation之间的TED'''
    treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''
    expr_expr_reverse_dict = {x: i for i, x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        '''
        [i, j] in treedis_matrix 分别指向两个expression在expr_id_dict的index
        下方公式是论文中Sim_eq(E1, E2)的计算公式，那说明d[1]存的是TED(E1, E2)
        到此为止treedis_data的结构就知晓了['E1;E2', TED(E1, E2)]
        '''
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)
    '''filter_ids存储所有length<10的expression在expr_id_dict中对应的id'''
    filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
    # treedis_matrix1 = copy.deepcopy(treedis_matrix)
    treedis_matrix2 = copy.deepcopy(treedis_matrix)
    for i in range(len(treedis_matrix)):
        '''negative infinity， 每个expression和其自身的distance应该为无穷小'''
        treedis_matrix2[i][i] = -float('inf')
    # for idx in filter_ids:
    #     treedis_matrix1[:, idx] = -float('inf')
    #     treedis_matrix1[idx, :] = -float('inf')
    for idx in filter_ids:
        '''有点奇怪，所有length<10的expression和其他expression之间的的distance记为无穷小'''
        treedis_matrix2[:, idx] = -float('inf')
        treedis_matrix2[idx, :] = -float('inf')

    res = []

    for d in train_data:
        postfix_positive = ' '.join(d['postfix'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        # if len(positive_ids) == 1:
        #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
        #     positive_ids = expr_id_dict[postfix_positive]
        minscore = float('inf')
        positive = d['id']
        for idx in positive_ids:
            '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
            if idx != d['id'] and sim(src, cached[idx]) < minscore:
                minscore = sim(src, cached[idx])
                positive = idx
        '选出和当前expression具有最高TED的expr，值得注意的是由于treedis_matrix2[i][i]==无穷小，所以选出来的实际上是次高值'
        # list[] => max index => 多个最大值则返回第一个
        # postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]

        max_indices = np.where(treedis_matrix2[exprpos] == np.max(treedis_matrix2[exprpos]))[0]
        random_index = np.random.choice(max_indices)
        postfix_negative = expr_expr_reverse_dict[random_index]

        negative_ids = expr_id_dict[postfix_negative]
        maxscore = -float('inf')
        negative = d['id']
        for idx in negative_ids:
            '选出具有最大Sim_BiBLEU的expr'
            if idx != d['id'] and sim(src, cached[idx]) > maxscore:
                maxscore = sim(src, cached[idx])
                negative = idx
        d['positive'] = positive
        d['negative'] = negative
        res.append(d)
        if len(res) % 100 == 0:
            print(len(res)/len(train_data))


    f = open(data_root_path + save_file, 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()


def preprocess_cl_RecursiveTreeDis_PosNN(data_root_path, train_file, treedis_file, save_file):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"seed is {seed}")
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + train_file)
    '''(TED)'''
    treedis_data = load_data(data_root_path + treedis_file)
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        # d['posifix'] = d['postfix_treedis']
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix_normed']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix_normed'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix_normed'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''建立全零矩阵，保存equation-equation之间的TED'''
    treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''
    expr_expr_reverse_dict = {x: i for i, x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        '''
        [i, j] in treedis_matrix 分别指向两个expression在expr_id_dict的index
        下方公式是论文中Sim_eq(E1, E2)的计算公式，那说明d[1]存的是TED(E1, E2)
        到此为止treedis_data的结构就知晓了['E1;E2', TED(E1, E2)]
        '''
        # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)
        # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] * (1 + np.log2(1 + abs(len1//2 - len2//2)))
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1]

    '''filter_ids存储所有length<10的expression在expr_id_dict中对应的id'''

    # filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
    treedis_matrix2 = copy.deepcopy(treedis_matrix)
    for i in range(len(treedis_matrix)):
        '''negative infinity， 每个expression和其自身的distance应该为无穷小
            新算法中应该改为无穷大'''
        treedis_matrix2[i][i] = -float('inf')
        # treedis_matrix2[i][i] = float('inf')

    # for idx in filter_ids:
    #     '''有点奇怪，所有出现频次<10的expression和其他expression之间的的distaance记为无穷小
    #         这样做会导致频次<10的expression再treedis_matrix2中被记录为无穷小，导致其永远无法被选为负样本
    #     '''
    #     treedis_matrix2[:, idx] = -float('inf')
    #     treedis_matrix2[idx, :] = -float('inf')

    res = []

    for d in train_data:
        postfix_positive = ' '.join(d['postfix_normed'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        if len(positive_ids) > 10:  # 如果positive_ids > 10则随机抽样增加多样性
            positive_ids = random.sample(positive_ids, 10)
        # if len(positive_ids) == 1:
        #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
        #     positive_ids = expr_id_dict[postfix_positive]
        minscore = float('inf')
        positive = d['id']
        for idx in positive_ids:
            '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
            if idx != d['id'] and sim(src, cached[idx]) < minscore:
                minscore = sim(src, cached[idx])
                positive = idx
        d['positive'] = positive
        '选出和当前expression具有最高TED的expr，值得注意的是由于treedis_matrix2[i][i]==无穷小，所以选出来的实际上是次高值'
        '从可选neg_postfix_index中随机抽样'
        # 最接近的expression如果有多个则随机抽样
        random_max_index = random.choice(np.where(treedis_matrix2[exprpos] == np.max(treedis_matrix2[exprpos]))[0])
        postfix_negative = expr_expr_reverse_dict[random_max_index]
        # postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]
        negative_ids = expr_id_dict[postfix_negative]
        # if len(negative_ids) > 10:
        #     negative_ids = random.sample(negative_ids, 10)
        maxscore = -float('inf')
        negative = d['id']
        for idx in negative_ids:
            '选出具有最大Sim_BiBLEU的expr'
            if idx != d['id'] and sim(src, cached[idx]) > maxscore:
                maxscore = sim(src, cached[idx])
                negative = idx
        d['negative'] = negative
        res.append(d)
        if len(res) % 100 == 0:
            print(len(res)/len(train_data))

    f = open(data_root_path + save_file, 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()



def preprocess_cl_PosN(data_root_path, train_file, treedis_file, save_file):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"seed is {seed}")
    print(data_root_path + 'is processing...')
    train_data = load_data(data_root_path + train_file)
    '''(TED)'''
    # treedis_data = load_data(data_root_path + treedis_file)
    cached = dict()
    expr_id_dict = dict()
    for d in train_data:
        # d['posifix'] = d['postfix_treedis']
        '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
        if ' '.join(d['postfix_normed']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix_normed'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix_normed'])].append(d['id'])
        '''cached保存id => text'''
        cached[d['id']] = d['text']
    '''postfix: i (i in expr_id_dict)'''
    expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
    '''postfix: i (i in expr_expr_dict)'''

    res = []
    count = 0
    for d in train_data:
        postfix_positive = ' '.join(d['postfix_normed'])
        '''exprpos 是当前postfix在expr_id_dict的index'''
        exprpos = expr_expr_dict[postfix_positive]
        src = d['text']
        'index_list'
        positive_ids = expr_id_dict[postfix_positive]
        if len(positive_ids) == 1:
            d['positive'] = d['positive']
        else:
            if len(positive_ids) > 10:  # 如果positive_ids > 10则随机抽样增加多样性
                positive_ids = random.sample(positive_ids, 10)
            # if len(positive_ids) == 1:
            #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
            #     positive_ids = expr_id_dict[postfix_positive]
            minscore = float('inf')
            positive = d['id']
            for idx in positive_ids:
                '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
                if idx != d['id'] and sim(src, cached[idx]) < minscore:
                    minscore = sim(src, cached[idx])
                    positive = idx

            d['positive'] = positive
            count += 1
        res.append(d)
        if len(res) % 100 == 0:
            print(len(res)/len(train_data))

    f = open(data_root_path + save_file, 'w', encoding='utf-8')
    print('replaced positive: ', count)
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

def preprocess_cl_PosNxNegN(data_root_path, train_file, treedis_file, save_file):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        print(f"seed is {seed}")
        print(data_root_path + 'is processing...')
        train_data = load_data(data_root_path + train_file)
        '''(TED)'''
        # treedis_data = load_data(data_root_path + treedis_file)
        cached = dict()
        expr_id_dict = dict()
        for d in train_data:
            # d['posifix'] = d['postfix_treedis']
            '''expr_id_dict保存postfix => id_list，每个postfix可能对应多个id'''
            if ' '.join(d['postfix']) not in expr_id_dict:
                expr_id_dict[' '.join(d['postfix'])] = [d['id']]
            else:
                expr_id_dict[' '.join(d['postfix'])].append(d['id'])
            '''cached保存id => text'''
            cached[d['id']] = d['text']
        '''postfix: i (i in expr_id_dict)'''
        expr_expr_dict = {x: i for i, x in enumerate(expr_id_dict.keys())}
        '''postfix: i (i in expr_expr_dict)'''

        res = []
        count = 0
        for d in train_data:
            postfix_positive = ' '.join(d['postfix'])
            '''exprpos 是当前postfix在expr_id_dict的index'''
            exprpos = expr_expr_dict[postfix_positive]
            src = d['text']
            'index_list'
            positive_ids = expr_id_dict[postfix_positive]
            if len(positive_ids) == 1:
                d['positive'] = d['positive']
            else:
                if len(positive_ids) > 10:  # 如果positive_ids > 10则随机抽样增加多样性
                    positive_ids = random.sample(positive_ids, 10)
                # if len(positive_ids) == 1:
                #     postfix_positive = expr_expr_reverse_dict[np.argmax(treedis_matrix1[exprpos])]
                #     positive_ids = expr_id_dict[postfix_positive]
                minscore = float('inf')
                positive = d['id']
                for idx in positive_ids:
                    '在所有具有相同expression（因为采用Exact Match）找出Sim_BiBLEU最小的expression'
                    if idx != d['id'] and sim(src, cached[idx]) < minscore:
                        minscore = sim(src, cached[idx])
                        positive = idx

                d['positive'] = positive
                count += 1
            res.append(d)
            if len(res) % 100 == 0:
                print(len(res) / len(train_data))

        f = open(data_root_path + save_file, 'w', encoding='utf-8')
        print('replaced positive: ', count)
        for d in res:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
        f.close()


