import argparse
import os.path

import torch
import random

from collections import Counter
from transformers import BertTokenizer, AutoConfig, AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup

from configuration.config import *
from models.text import Encoder
from models.tree import TreeDecoder
from models.train_and_evaluate import Solver, train_double, evaluate_double, cl_loss
from preprocess.tuple import generate_tuple, convert_tuple_to_id
from preprocess.metric import compute_prefix_tree_result
from utils.logger import *

'''English: https://huggingface.co/bert-base-uncased, 
    Chinese: https://huggingface.co/yechen/bert-base-chinese'''


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def set_seed(seed=1):
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


def train(data_root_path, train_file, save_path, gpu, epochs, current_epoch=0):
    set_seed()
    data_root_path = data_root_path
    train_data = load_data(data_root_path + train_file)
    test_data = load_data(data_root_path + 'dev.jsonl')

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    config = AutoConfig.from_pretrained(pretrain_model_path)
    tokens = Counter()
    max_nums_len = 0
    for d in train_data + test_data:
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
        src = tokenizer.encode('<O>' + d['text'], max_length=max_text_len)
        tgt1 = [tokens_dict1.get(x, len(op_tokens)) for x in d['prefix']]
        tgt2 = generate_tuple([d['postfix']], op_tokens)
        tgt2 = convert_tuple_to_id(tgt2, op_dict2, constant_dict2, source_dict2)
        num = []

        for i, s in enumerate(src):
            if s in number_tokens_ids:
                num.append(i)

        if len(num) != len(d['nums']):
            print(d)

        # assert len(num) == len(d['nums']), "数字个数不匹配！%s vs %s" % (len(num), len(d['nums']))
        value = [x for x in d['nums']]
        train_batches1.append((src, num, value, tgt1, tgt2))
        cached[d['id']] = (src, num, value, tgt1, tgt2)

    train_batches = []
    for d in train_data:
        train_batches.append([cached[d['id']], cached[d['positive']], cached[d['negative']]])

    dev_batches = []
    for d in test_data:
        src1 = tokenizer.encode('<O>' + d['text'], max_length=max_text_len)
        num = []
        for i, s in enumerate(src1):
            if s in number_tokens_ids:
                num.append(i)
        value = [x for x in d['nums']]
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
        batches2 = []
        batches3 = []
        for pair in pairs:
            text_ids, num_ids, graphs, equ_ids, tuple_ids = [], [], [], [], []
            max_text = max([len(x[0][0]) for x in pair])
            max_num = max([len(x[0][1]) for x in pair])
            max_equ = max([len(x[0][3]) for x in pair])
            max_tuple = max([len(x[0][4]) for x in pair])
            for _, p in enumerate(pair):
                text, num, value, equ, tuple = p[0]
                text_ids.append(text + [tokenizer.pad_token_id] * (max_text - len(text)))
                num_ids.append(num + [-1] * (max_num - len(num)))
                graphs.append(generate_graph(max_num, value))
                equ_ids.append(equ + [-1] * (max_equ - len(equ)))
                tuple_ids.append(tuple + [[-1, -1, -1, -1, -1]] * (max_tuple - len(tuple)))
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
            batches1.append((text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids))

            text_ids, num_ids, graphs, equ_ids, tuple_ids = [], [], [], [], []
            max_text = max([len(x[1][0]) for x in pair])
            max_num = max([len(x[1][1]) for x in pair])
            max_equ = max([len(x[1][3]) for x in pair])
            max_tuple = max([len(x[1][4]) for x in pair])
            for _, p in enumerate(pair):
                text, num, value, equ, tuple = p[1]
                text_ids.append(text + [tokenizer.pad_token_id] * (max_text - len(text)))
                num_ids.append(num + [-1] * (max_num - len(num)))
                graphs.append(generate_graph(max_num, value))
                equ_ids.append(equ + [-1] * (max_equ - len(equ)))
                tuple_ids.append(tuple + [[-1, -1, -1, -1, -1]] * (max_tuple - len(tuple)))
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
            batches2.append((text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids))

            text_ids, num_ids, graphs, equ_ids, tuple_ids = [], [], [], [], []
            max_text = max([len(x[2][0]) for x in pair])
            max_num = max([len(x[2][1]) for x in pair])
            max_equ = max([len(x[2][3]) for x in pair])
            max_tuple = max([len(x[2][4]) for x in pair])
            for _, p in enumerate(pair):
                text, num, value, equ, tuple = p[2]
                text_ids.append(text + [tokenizer.pad_token_id] * (max_text - len(text)))
                num_ids.append(num + [-1] * (max_num - len(num)))
                graphs.append(generate_graph(max_num, value))
                equ_ids.append(equ + [-1] * (max_equ - len(equ)))
                tuple_ids.append(tuple + [[-1, -1, -1, -1, -1]] * (max_tuple - len(tuple)))
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
            batches3.append((text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids))
        return batches1, batches2, batches3

    pretrain_model = AutoModel.from_pretrained(pretrain_model_path)
    pretrain_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrain_model)
    treedecoder = TreeDecoder(config, len(op_tokens), len(constant_tokens), embedding_size)

    solver = Solver(encoder, treedecoder)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    solver.to(device)
    batches1, batches2, batches3 = data_generator(train_batches, batch_size)
    optimizer = AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
    global_steps = len(batches1) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=global_steps * 0.1,
                                                num_training_steps=global_steps)

    # train
    solver.zero_grad()

    best_acc = 0
    epochs += current_epoch
    for e in range(current_epoch, epochs):
        log = open(save_path + '_log.txt', 'a')
        print("epoch:", e)
        solver.train()
        loss_total = 0.0
        random.shuffle(train_batches)
        batches1, batches2, batches3 = data_generator(train_batches, batch_size)
        bar = tqdm(range(len(batches1)), total=len(batches1))
        for i in bar:
            batch1, batch2, batch3 = batches1[i], batches2[i], batches3[i]
            batch1 = [_.to(device) for _ in batch1]
            batch2 = [_.to(device) for _ in batch2]
            batch3 = [_.to(device) for _ in batch3]

            text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids = batch1
            loss1, encoded1 = train_double(solver, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, op_tokens,
                                           constant_tokens)
            text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids = batch2
            encoded2 = solver.encoder(text_ids, text_pads, num_ids, num_pads)
            text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids = batch3
            encoded3 = solver.encoder(text_ids, text_pads, num_ids, num_pads)
            'encoded1/2/3: anchor data/ positive/ negative '
            'encoded1[text]: batch_size x text_length x embedding_size (e.g. 16*66*768)'
            'encoded1[text][:, 0]: batch_size x embedding_size'
            loss2 = cl_loss(encoded1['text'][:, 0], encoded2['text'][:, 0], encoded3['text'][:, 0], temperature)
            loss = loss1 + alpha * loss2
            # loss = loss1
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_total /= len(batches1)
        log.write("epoch:" + str(e) + "\tloss:" + str(loss_total) + "\n")
        logger.info(f"epoch: {e} - loss: {loss_total}")

        if (e >= 60 and e % 5 == 0) or e >= 110:
        # if True:
            solver.eval()
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            all_results = []
            correct_results = []
            wrong_results = []
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
                                            op_tokens, constant_tokens, max_equ_len, beam_size=3)
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
                temp = {}
                temp['id'] = d['id']
                temp['text'] = d['text']
                temp['nums'] = d['nums']
                temp['ans'] = d['answer']
                temp['prefix'] = d['prefix']
                temp['postfix'] = d['postfix']
                temp['tree_out1'] = tree_out1
                temp['tree_score1'] = tree_score1
                temp['tree_result1'] = tree_val_ac1
                if val_ac:
                    correct_results.append(temp)
                else:
                    wrong_results.append(temp)
                all_results.append(temp)

            if float((value_ac) / eval_total) > best_acc:
                best_acc = float((value_ac) / eval_total)
                f = open(save_path + '/correct_result_test.jsonl', 'w', encoding='utf-8')
                for d in correct_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open(save_path + '/wrong_results_test.jsonl', 'w', encoding='utf-8')
                for d in wrong_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open(save_path + '/all_results_test.jsonl', 'w', encoding='utf-8')
                for d in all_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                try:
                    os.removedirs(save_path + '/models_best')
                    solver.save_pretrained(save_path + '/models_best')
                    tokenizer.save_pretrained(save_path + '/models_best')
                    log.write("model updated")
                    logger.info("model updated")
                except:
                    solver.save_pretrained(save_path + '/models_best')
                    tokenizer.save_pretrained(save_path + '/models_best')
                    log.write("model saved")
                    logger.info("model saved")

            log.write("epoch:" + str(e) + "\tequ_acc:" + str(float(equation_ac) / eval_total) + "\tval_acc:" + str(
                float(value_ac) / eval_total) + "\n")
            log.close()
            logger.info(
                f"epoch: {e} - equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")
        torch.cuda.empty_cache()
    solver.save_pretrained(save_path + '/models_last')
    tokenizer.save_pretrained(save_path + '/models_last')

    folds_scores.append(best_acc)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='Train_Math23k', type=str)
    parser.add_argument('--device', default=0, type=int)  # WARNING: DO NOT USE DEVICES=[4, 5]!!!
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    # parser.add_argument('--tokenizer', default='bert-wwm', type=str, choices=['bert', 'bert-wwm', 'roberta-wwm'])
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--train_file', default='train.jsonl', type=str)

    return parser


if __name__ == '__main__':
    'AsDiv-A 5CV'
    # pretrain_model_path = 'bert-base-uncased'
    # run_name = f'Train_AsDiv-A'
    # train_file = 'train_cl.jsonl'
    # data_name = 'asdiv-a'
    # discription = 'AsDiv-A w/CL bert-encoder, epoch50'
    # cl = 'Y'
    # train_set = 'asdiv-a'
    # results_path = f'./results/{data_name}.jsonl'
    # folds_scores = []
    #
    # for i in range(5):
    #     '改数据集！！！！！'
    #     data_root_path = f'data/AsDiv-A_5cv/fold{i}/'
    #     checkpoints_path = f'./checkpoints/asdiv-a/{run_name}/fold{i}/'
    #     print(run_name + f"fold{i}")
    #     if not os.path.exists(checkpoints_path):
    #         os.makedirs(checkpoints_path)
    #     train(data_root_path, train_file, checkpoints_path, gpu=gpu_id, epochs=50, current_epoch=0)
    # acc_avg = sum(folds_scores) / len(folds_scores)
    # print('Overall accuracy: ', acc_avg)

    max_text_len = 256
    max_equ_len = 45
    embedding_size = 128

    parser = get_parser()
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    temperature = args.temperature
    alpha = args.alpha

    'Math23k'
    # pretrain_model_path = 'yechen/bert-base-chinese'
    # pretrain_model_path = 'bert-base-uncased'
    # gpu_id = args.device
    # run_name = args.run_name
    # train_file = args.train_file
    # data_name = 'svamp'
    # data_root_path = f'../data/asdiv-a_mawps_svamp/'
    # # train_set = 'math23k'
    # folds_scores = []
    #
    # checkpoints_path = f'./checkpoints/{run_name}/'
    # results_path = f'./results/{data_name}.jsonl'
    #
    # print(run_name)
    # if not os.path.exists(checkpoints_path):
    #     os.makedirs(checkpoints_path)
    # train(data_root_path, train_file, checkpoints_path, gpu=gpu_id, epochs=120, current_epoch=0)

    'MathQA'
    # pretrain_model_path = 'bert-base-uncased'
    # gpu_id = args.device
    # run_name = args.run_name
    # train_file = args.train_file
    # data_name = 'svamp'
    # data_root_path = f'../data/mathqa/'
    # folds_scores = []
    #
    # checkpoints_path = f'./checkpoints/{run_name}/'
    # results_path = f'./results/{data_name}.jsonl'
    #
    # print(run_name)
    # if not os.path.exists(checkpoints_path):
    #     os.makedirs(checkpoints_path)
    # train(data_root_path, train_file, checkpoints_path, gpu=gpu_id, epochs=args.epoch, current_epoch=0)

    'AsDiv-A 5CV'
    pretrain_model_path = 'bert-base-uncased'
    gpu_id = args.device
    run_name = args.run_name
    train_file = args.train_file
    # data_name = 'asdiv'
    data_root_path = f'../data/cv_asdiv-a/'
    folds_scores = []

    checkpoints_path = f'./checkpoints/{run_name}/'
    # results_path = f'./results/{data_name}.jsonl'
    print(run_name)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    for i in range(5):
        if not os.path.exists(os.path.join(checkpoints_path, f'fold{i}')):
            os.makedirs(os.path.join(checkpoints_path, f'fold{i}'))
        train(os.path.join(data_root_path, f'fold{i}/'), train_file, os.path.join(checkpoints_path, f'fold{i}'),
              gpu=gpu_id, epochs=args.epoch, current_epoch=0)