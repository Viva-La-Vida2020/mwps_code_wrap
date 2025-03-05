import argparse
import torch
import random
from transformers import AutoConfig, AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup
from configuration.config import *
from models.text import Encoder
from models.tree import TreeDecoder
from models.train_and_evaluate import Solver, train_double, evaluate_double, cl_loss
from preprocess.metric import compute_prefix_tree_result
from utils.logger import *
from data.data_module_gts import MathDataModule
'''English: https://huggingface.co/bert-base-uncased, 
    Chinese: https://huggingface.co/yechen/bert-base-chinese'''


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('seed:', seed)


def train(data_root_path, train_file, save_path, gpu, epochs, current_epoch=0):
    set_seed()

    data_module = MathDataModule(
        data_dir="../data/mathqa",  # Change this to your dataset directory
        train_file="train.jsonl",  # Ensure this file exists in `data/`
        test_file="test.jsonl",  # Ensure this file exists in `data/`
        tokenizer_path="../pretrained_model/bert-base-chinese",  # Change if using another tokenizer
        batch_size=16,  # Small batch size for easy debugging
        max_text_len=256
    )

    # Setup data (mimics Lightning's internal workflow)
    data_module.setup()

    # Get the train dataloader
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    config = AutoConfig.from_pretrained(pretrain_model_path)
    tokenizer = data_module.tokenizer
    op_tokens = data_module.op_tokens
    constant_tokens = data_module.constant_tokens
    pretrain_model = AutoModel.from_pretrained(pretrain_model_path)
    pretrain_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrain_model)
    treedecoder = TreeDecoder(config, len(op_tokens), len(constant_tokens), embedding_size)

    solver = Solver(encoder, treedecoder)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    solver.to(device)
    # batches1 = data_generator(train_batches, batch_size)
    optimizer = AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
    # global_steps = len(batches1) * epochs
    global_steps = len(train_loader) * epochs
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

        # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epochs + 1}")
        #
        # for i, (batch) in progress_bar:
        #     text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads = (
        #         batch[k].to(device) for k in ["text_ids", "text_pads", "num_ids", "num_pads", "equ_ids", "equ_pads"]
        #     )
        #     loss1, encoded1 = train_double(solver, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, op_tokens,
        #                                    constant_tokens)
        #     loss2 = 0
        #     loss = loss1 + alpha * loss2
        #     # loss = loss1
        #     loss_total += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
        #     if (i + 1) % 100 == 0:
        #         torch.cuda.empty_cache()
        #
        # loss_total /= len(train_loader)
        # log.write("epoch:" + str(e) + "\tloss:" + str(loss_total) + "\n")
        # logger.info(f"epoch: {e} - loss: {loss_total}")

        # if (e >= 60 and e % 5 == 0) or e >= 110:
        if True:
            solver.eval()
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            # bar = tqdm(enumerate(test_loader), total=len(test_loader))
            # for _, (text1, num, value, d) in bar:
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (batch) in progress_bar:
                text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads = (
                    batch[k].to(device) for k in ["text_ids", "text_pads", "num_ids", "num_pads", "equ_ids", "equ_pads"]
                )
                tree_res1 = evaluate_double(solver, text_ids, text_pads, num_ids, num_pads,
                                            op_tokens, constant_tokens, max_equ_len, beam_size=3)
                tree_out1, tree_score1 = tree_res1.out, tree_res1.score
                ids_dict = data_module.id_dict
                tree_out1 = [ids_dict[x] for x in tree_out1]
                tree_val_ac1, tree_equ_ac1, _, _ = compute_prefix_tree_result(tree_out1, batch['prefix'], batch['answer'],
                                                                              batch['nums'])
                scores = [tree_score1]
                score_index = np.array(scores).argmax()
                if score_index == 0:
                    val_ac = tree_val_ac1
                    equ_ac = tree_equ_ac1
                value_ac += val_ac
                equation_ac += equ_ac
                eval_total += 1
            logger.info(
                f"epoch: {e} - equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # solver.save_pretrained(save_path + '/models_last')
    # tokenizer.save_pretrained(save_path + '/models_last')

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
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--train_file', default='train.jsonl', type=str)

    return parser


if __name__ == '__main__':
    max_text_len = 256
    max_equ_len = 45
    embedding_size = 128

    parser = get_parser()
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    temperature = args.temperature
    alpha = args.alpha

    'MathQA'
    pretrain_model_path = '../pretrained_model/bert-base-chinese'
    gpu_id = args.device
    run_name = args.run_name
    train_file = args.train_file
    data_name = 'mathqa'
    data_root_path = f'../data/mathqa/'
    folds_scores = []

    checkpoints_path = f'./checkpoints/{run_name}/'
    results_path = f'./results/{data_name}.jsonl'

    print(run_name)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    train(data_root_path, train_file, checkpoints_path, gpu=gpu_id, epochs=args.epoch, current_epoch=0)