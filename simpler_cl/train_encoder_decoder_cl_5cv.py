# -*- coding: utf-8 -*-
import argparse
import torch
import random
from collections import Counter
from transformers import BertTokenizer, AutoConfig, AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup
from configuration.config import *
from models_GTS.text import Encoder
from models_GTS.tree import TreeDecoder
from models_GTS.train_and_evaluate import Solver, train_double, evaluate_double, Subspace
from preprocess.tuple import generate_tuple, convert_tuple_to_id
from preprocess.metric import compute_prefix_tree_result

MAX_TEXT_LEN = 256
MAX_EQU_LEN = 60
EMBEDDING_SIZE = 128
EMBEDDING_BERT = 768
TEMPERATURE = 0.05


def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('seed:', seed)


def generate_graph(max_num_len, nums):
    diag_ele = np.ones(max_num_len)
    graph1 = np.diag(diag_ele)
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] <= nums[j]:
                graph1[i][j] = 1
            else:
                graph1[j][i] = 1
    graph2 = graph1.T
    return [graph1.tolist(), graph2.tolist()]


def train_simpler_cl(args, fold):
    set_seed(args.seed)
    # Load dataset
    if args.dataset in ['Math23k']:
        if args.tokenizer == 'bert':
            pretrain_model_path = "yechen/bert-base-chinese"
        elif args.tokenizer == 'bert-wwm':
            pretrain_model_path = "hfl/chinese-bert-wwm-ext"
            # pretrain_model_path = "../pretrained_model/chinese-bert-wwm-ext"
        elif args.tokenizer == 'roberta-wwm':
            pretrain_model_path = "hfl/chinese-roberta-wwm-ext"

        config = AutoConfig.from_pretrained(pretrain_model_path)
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        epochs = args.epoch
        data_root_path = '../data/math23k/'
        train_data = load_data(data_root_path + 'train.jsonl')
        # dev_data = load_data(data_root_path + 'Math23K_dev.jsonl')
        # test_data = load_data(data_root_path + 'Math23K_test.jsonl')
        dev_data = load_data(data_root_path + 'test.jsonl')
        test_data = load_data(data_root_path + 'test.jsonl')
    elif args.dataset in ['MathQA']:
        if args.tokenizer == 'bert':
            pretrain_model_path = "bert-base-uncased"

        config = AutoConfig.from_pretrained(pretrain_model_path)
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        epochs = args.epoch
        data_root_path = '../data/mathqa/'
        train_data = load_data(data_root_path + 'train_cl.jsonl')
        dev_data = load_data(data_root_path + 'test.jsonl')
        test_data = load_data(data_root_path + 'test.jsonl')
    elif args.dataset in ['AsDiv-A']:
        if args.tokenizer == 'bert':
            pretrain_model_path = "bert-base-uncased"

        config = AutoConfig.from_pretrained(pretrain_model_path)
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        epochs = args.epoch
        data_root_path = f'../data/cv_asdiv-a/fold{fold}/'
        train_data = load_data(data_root_path + 'train_cl.jsonl')
        dev_data = load_data(data_root_path + 'dev.jsonl')
        test_data = load_data(data_root_path + 'dev.jsonl')
    elif args.dataset in ['SVAMP']:
        if args.tokenizer == 'bert':
            pretrain_model_path = "bert-base-uncased"

        config = AutoConfig.from_pretrained(pretrain_model_path)
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        epochs = args.epoch
        data_root_path = '../data/asdiv-a_mawps_svamp/'
        train_data = load_data(data_root_path + 'train.jsonl')
        dev_data = load_data(data_root_path + 'test.jsonl')
        test_data = load_data(data_root_path + 'test.jsonl')

    tokens = Counter()
    max_nums_len = 0
    for d in train_data + dev_data + test_data:
        tokens += Counter(d['prefix'])
        max_nums_len = max(max_nums_len, len(d['nums']))

    tokens = list(tokens)
    op_tokens = [x for x in tokens if x[0].lower() != 'c' and x[0].lower() != 'n']
    constant_tokens = [x for x in tokens if x[0].lower() == 'c']
    number_tokens = ['N_' + str(x) for x in range(max_nums_len)]
    op_tokens.sort()
    constant_tokens = sorted(constant_tokens, key=lambda x: float(x[2:].replace('_', '.')))
    number_tokens = sorted(number_tokens, key=lambda x: int(x[2:]))
    mtokens = ['<O>', '<Add>', '<Mul>']
    tokens1 = op_tokens + constant_tokens + number_tokens
    tokens_dict1 = {x: i for i, x in enumerate(tokens1)}
    ids_dict1 = {x[1]: x[0] for x in tokens_dict1.items()}

    source_dict2 = {'C:': 0, 'N:': 1, 'M:': 2}
    op_dict2 = {x: i for i, x in enumerate(op_tokens)}
    op_dict2['<s>'] = len(op_dict2)
    op_dict2['</s>'] = len(op_dict2)
    constant_dict2 = {x: i for i, x in enumerate(constant_tokens)}

    tokenizer.add_special_tokens({'additional_special_tokens': number_tokens + mtokens})
    number_tokens_ids = [tokenizer.convert_tokens_to_ids(x) for x in number_tokens]
    number_tokens_ids = set(number_tokens_ids)

    train_batches1 = []
    cached = {}
    for d in train_data:
        src = tokenizer.encode('<O>' + d['text'], max_length=MAX_TEXT_LEN)
        tgt1 = [tokens_dict1.get(x, len(op_tokens)) for x in d['prefix']]

        tgt2 = generate_tuple([d['postfix']], op_tokens)

        tgt2 = convert_tuple_to_id(tgt2, op_dict2, constant_dict2, source_dict2)
        num = []
        for i, s in enumerate(src):
            if s in number_tokens_ids:
                num.append(i)
        assert len(num) == len(d['nums']), "Number count not match！%s vs %s" % (len(num), len(d['nums']))
        value = [eval(str(x)) for x in d['nums']]

        train_batches1.append(
            (src, num, value, tgt1, tgt2, d['postfix'], d['prefix'], d['root_nodes'], d['longest_view']))
        cached[d['id']] = (src, num, value, tgt1, tgt2, d['postfix'], d['prefix'], d['root_nodes'], d['longest_view'])

    train_batches = []
    for d in train_data:
        train_batches.append([cached[d['id']]])

    dev_batches = []
    for d in dev_data:
        src1 = tokenizer.encode('<O>' + d['text'], max_length=MAX_TEXT_LEN)
        num = []
        for i, s in enumerate(src1):
            if s in number_tokens_ids:
                num.append(i)
        value = [eval(x) for x in d['nums']]
        dev_batches.append((src1, num, value, d))

    def data_generator(train_batches, batch_size):
        i = 0
        pairs = []
        while i + batch_size < len(train_batches):
            pair = train_batches[i: i + batch_size]
            pairs.append(pair)
            i += batch_size
        # pairs.append(train_batches[i:])
        batches1 = []

        for pair in pairs:
            text_ids, num_ids, graphs, equ_ids, tuple_ids = [], [], [], [], []
            max_text = max([len(x[0][0]) for x in pair])
            max_num = max([len(x[0][1]) for x in pair])
            max_equ = max([len(x[0][3]) for x in pair])
            max_tuple = max([len(x[0][4]) for x in pair])
            postfix_list = []
            prefix_list = []
            root_nodes_list = []
            longest_view_list = []
            for _, p in enumerate(pair):
                text, num, value, equ, tuple, postfix, prefix, root_nodes, longest_view = p[0]
                text_ids.append(text + [tokenizer.pad_token_id] * (max_text - len(text)))
                num_ids.append(num + [-1] * (max_num - len(num)))
                graphs.append(generate_graph(max_num, value))
                equ_ids.append(equ + [-1] * (max_equ - len(equ)))
                tuple_ids.append(tuple + [[-1, -1, -1, -1, -1]] * (max_tuple - len(tuple)))
                postfix_list.append(postfix)
                prefix_list.append(prefix)
                root_nodes_list.append(root_nodes)
                longest_view_list.append(longest_view)
            text_ids = torch.tensor(text_ids, dtype=torch.long)
            num_ids = torch.tensor(num_ids, dtype=torch.long)
            graphs = torch.tensor(graphs, dtype=torch.float)
            equ_ids = torch.tensor(equ_ids, dtype=torch.long)
            tuple_ids = torch.tensor(tuple_ids, dtype=torch.long)
            text_pads = text_ids != tokenizer.pad_token_id
            text_pads = text_pads.float()
            num_pads = num_ids != -1
            num_pads = num_pads.float()
            equ_pads = equ_ids != -1
            equ_ids[~equ_pads] = 0
            equ_pads = equ_pads.float()

            batches1.append((text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids, postfix_list,
                             prefix_list, root_nodes_list, longest_view_list))

        return batches1

    pretrain_model = AutoModel.from_pretrained(pretrain_model_path)
    pretrain_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrain_model)
    treedecoder = TreeDecoder(config, len(op_tokens), len(constant_tokens), EMBEDDING_SIZE)
    subspace = Subspace(EMBEDDING_BERT, EMBEDDING_SIZE, len_subspace=3)  # holistic, root, longest_path
    solver = Solver(encoder, treedecoder, subspace)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver.to(device)
    batches1 = data_generator(train_batches, args.batch_size)
    optimizer = AdamW(solver.parameters(), lr=args.lr, weight_decay=0.01)
    global_steps = len(batches1) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=global_steps * 0.1,
                                                num_training_steps=global_steps)

    # train
    solver.zero_grad()

    if not os.path.exists(args.save):
        os.mkdir(args.save)
    save_dir = os.path.join(args.save, f'{args.dataset}_GTS_{args.CL}_{args.similarity}_BS{args.batch_size}_Epoch{args.epoch}_Seed{args.seed}_{args.tokenizer}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(args.save, f'{args.dataset}_GTS_{args.CL}_{args.similarity}_BS{args.batch_size}_Epoch{args.epoch}_Seed{args.seed}_{args.tokenizer}/fold{fold}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # criterion = SupConLoss.SupConLoss()
    best_val_acc = 0

    for e in range(epochs):
        log = open(os.path.join(save_dir, 'log.txt'), 'a')
        print("epoch:", e)
        solver.train()
        loss_total = 0.0
        loss_ce_total = 0.0
        loss_c1_total = 0.0
        random.shuffle(train_batches)
        batches1 = data_generator(train_batches, args.batch_size)

        bar = tqdm(range(len(batches1)), total=len(batches1))
        for i in bar:
            batch1 = batches1[i]
            postfix = batch1[-4]
            prefix = batch1[-3]
            root_nodes = batch1[-2]
            longest_view = batch1[-1]

            batch1 = [_.to(device) for _ in batch1[:-4]]

            text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids = batch1
            loss1, loss_c1 = train_double(args, solver, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads,
                                          op_tokens, constant_tokens, postfix, prefix, root_nodes, longest_view, )
            loss = loss1 + loss_c1

            loss_total += loss.item()
            loss_ce_total += loss1.item()
            loss_c1_total += loss_c1.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_total /= len(batches1)
        loss_ce_total /= len(batches1)
        loss_c1_total /= len(batches1)
        logger.info(f"epoch: {e} - loss: {loss_total}- loss CE: {loss_ce_total}  - loss cl: {loss_c1_total}")
        if e >= 10:
            solver.eval()

            value_ac = 0
            equation_ac = 0
            eval_total = 0
            best_epoch = 1
            bar = tqdm(enumerate(dev_batches), total=len(dev_batches))
            for _, (text1, num, value, d) in bar:
                text1_ids = torch.tensor([text1], dtype=torch.long)
                num_ids = torch.tensor([num], dtype=torch.long)
                graphs = generate_graph(len(num), value)
                graphs = torch.tensor([graphs], dtype=torch.float)
                text_pads = text1_ids != tokenizer.pad_token_id
                text_pads = text_pads.float()
                num_pads = num_ids != -1
                num_pads = num_pads.float()
                batch = [text1_ids, text_pads, num_ids, num_pads, graphs]
                batch = [_.to(device) for _ in batch]
                text1_ids, text_pads, num_ids, num_pads, graphs = batch
                tree_res1 = evaluate_double(solver, text1_ids, text_pads, num_ids, num_pads,
                                            op_tokens, constant_tokens, MAX_EQU_LEN, beam_size=3)
                tree_out1, tree_score1 = tree_res1.out, tree_res1.score
                tree_out1 = [ids_dict1[x] for x in tree_out1]
                tree_val_ac1, tree_equ_ac1, _, _ = compute_prefix_tree_result(tree_out1, d['prefix'], d['answer'],
                                                                              d['nums'])
                scores = [tree_score1]
                score_index = np.array(scores).argmax()
                if score_index == 0:
                    val_ac = tree_val_ac1
                    equ_ac = tree_equ_ac1
                value_ac += val_ac
                equation_ac += equ_ac
                eval_total += 1

            log.write("epoch:" + str(e)+ "\tequ_acc:" + str(float(equation_ac) / eval_total) + "\tval_acc:" + str(float(value_ac) / eval_total) + "\n")
            logger.info(
                f"epoch: {e} - equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")
            if float(value_ac) / eval_total > best_val_acc:
                best_val_acc = float(value_ac) / eval_total
                log.write(f"epoch: {e} - current best accuracy: {best_val_acc}\n")
                logger.info(f"epoch: {e} - current best accuracy: {best_val_acc}")
                best_epoch = e
                solver.save_pretrained(os.path.join(save_dir, 'models_best'))
                tokenizer.save_pretrained(os.path.join(save_dir, 'models_best'))

    return best_val_acc, best_epoch


def test(args):
    config = AutoConfig.from_pretrained(args.ckpt)
    tokenizer = BertTokenizer.from_pretrained(args.ckpt)
    data_root_path = 'data/Math23k/'
    train_data = load_data(data_root_path + 'Math23K_train.jsonl')
    dev_data = load_data(data_root_path + 'Math23K_dev.jsonl')
    test_data = load_data(data_root_path + 'Math23K_test.jsonl')

    tokens = Counter()
    max_nums_len = 0
    for d in train_data + dev_data + test_data:
        tokens += Counter(d['prefix'])
        max_nums_len = max(max_nums_len, len(d['nums']))

    tokens = list(tokens)
    op_tokens = [x for x in tokens if x[0].lower() != 'c' and x[0].lower() != 'n']
    constant_tokens = [x for x in tokens if x[0].lower() == 'c']
    number_tokens = ['N_' + str(x) for x in range(max_nums_len)]
    op_tokens.sort()
    constant_tokens = sorted(constant_tokens, key=lambda x: float(x[2:].replace('_', '.')))
    number_tokens = sorted(number_tokens, key=lambda x: int(x[2:]))
    tokens1 = op_tokens + constant_tokens + number_tokens
    tokens_dict1 = {x: i for i, x in enumerate(tokens1)}
    ids_dict1 = {x[1]: x[0] for x in tokens_dict1.items()}

    op_dict2 = {x: i for i, x in enumerate(op_tokens)}
    op_dict2['<s>'] = len(op_dict2)
    op_dict2['</s>'] = len(op_dict2)

    number_tokens_ids = [tokenizer.convert_tokens_to_ids(x) for x in number_tokens]
    number_tokens_ids = set(number_tokens_ids)

    test_batches = []
    for d in test_data:
        src1 = tokenizer.encode('<O>' + d['text'], max_length=MAX_TEXT_LEN)
        num = []
        for i, s in enumerate(src1):
            if s in number_tokens_ids:
                num.append(i)
        value = [eval(x) for x in d['nums']]
        test_batches.append((src1, num, value, d))

    pretrain_model = AutoModel.from_pretrained(args.ckpt)
    pretrain_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrain_model)
    treedecoder = TreeDecoder(config, len(op_tokens), len(constant_tokens), EMBEDDING_SIZE)
    treedecoder.load_state_dict(torch.load(os.path.join(args.ckpt, 'decoder1.pt')))
    subspace = Subspace(EMBEDDING_BERT, EMBEDDING_SIZE, len_subspace=3)  # holistic, root, longest_path
    subspace.load_state_dict(torch.load(os.path.join(args.ckpt, 'Subspace.pt')))
    solver = Solver(encoder, treedecoder, subspace)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver.to(device)
    solver.eval()

    value_ac = 0
    equation_ac = 0
    eval_total = 0

    bar = tqdm(enumerate(test_batches), total=len(test_batches))
    for _, (text1, num, value, d) in bar:
        text1_ids = torch.tensor([text1], dtype=torch.long)
        num_ids = torch.tensor([num], dtype=torch.long)
        graphs = generate_graph(len(num), value)
        graphs = torch.tensor([graphs], dtype=torch.float)
        text_pads = text1_ids != tokenizer.pad_token_id
        text_pads = text_pads.float()
        num_pads = num_ids != -1
        num_pads = num_pads.float()
        batch = [text1_ids, text_pads, num_ids, num_pads, graphs]
        batch = [_.to(device) for _ in batch]
        text1_ids, text_pads, num_ids, num_pads, graphs = batch
        tree_res1 = evaluate_double(solver, text1_ids, text_pads, num_ids, num_pads,
                                    op_tokens, constant_tokens, MAX_EQU_LEN, beam_size=3)
        tree_out1, tree_score1 = tree_res1.out, tree_res1.score
        tree_out1 = [ids_dict1[x] for x in tree_out1]
        tree_val_ac1, tree_equ_ac1, _, _ = compute_prefix_tree_result(tree_out1, d['prefix'], d['answer'],
                                                                      d['nums'])
        scores = [tree_score1]
        score_index = np.array(scores).argmax()
        if score_index == 0:
            val_ac = tree_val_ac1
            equ_ac = tree_equ_ac1
        value_ac += val_ac
        equation_ac += equ_ac
        eval_total += 1

    logger.info(
        f"equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='./experiments', type=str)
    parser.add_argument('--device', default=0, type=int)  #WARNING: DO NOT USE DEVICES=[4, 5]!!!
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--tokenizer', default='bert-wwm', type=str, choices=['bert', 'bert-wwm', 'roberta-wwm'])
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--dataset', default='Math23k', type=str, choices=['Math23k', 'AsDiv-A', 'SVAMP', 'MathQA'])
    parser.add_argument('--CL', default='SimplerCL', type=str, choices=['SimplerCL', 'SimCLR', 'NoCL'])
    parser.add_argument('--similarity', default='TLWD', type=str, choices=['TLWD', 'TED'])
    parser.add_argument('--H', action='store_true', default=True, help='CL from Holistic View')
    parser.add_argument('--P', action='store_true', default=True, help='CL from Primary View')
    parser.add_argument('--L', action='store_true', default=True, help='CL from Longest View')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--vis', action='store_true', help='Save embeddings for TSNE-Visulization')
    # parser.add_argument('--ckpt', default='./experiments/Math23k_GTS/models_best', type=str)#experiments/Math23k_GTS_TLWD/models_bestn#./ckpts/models_best
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    if args.train:
        for i in range(5):
            train_simpler_cl(args, fold=i)
    elif args.test:
        print('Test Begin')
        test(args)
