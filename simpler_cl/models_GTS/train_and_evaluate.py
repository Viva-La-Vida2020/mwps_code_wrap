import copy
import torch
import torch.nn as nn
from .tree import TreeNode, TreeEmbedding, TreeBeam, copy_list
from .sup_con_loss import SupConLoss
from zss import simple_distance, Node
import numpy as np
import torch.nn.functional as F
from .prep_tlwd import calculate_tlwd


def from_postfix_to_tree(postfix, no_num):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            # st.append(Node(p))
            if no_num:
                st.append(Node("N"))
            else:
                st.append(Node(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
    return st.pop()


def tlwd_score_multiview(prefix, no_num=False):
    def get_view_score(prefix, no_num, alpha):
        expr = []
        for d in prefix:
            expr.append(' '.join(d))
        score = np.zeros((len(expr), len(expr)))
        for i in range(len(expr)):
            for j in range(i, len(expr)):
                sim_tlwd = calculate_tlwd(expr[i], expr[j], alpha, no_num)
                score[i, j] = sim_tlwd
                score[j, i] = sim_tlwd
        min_value = np.min(score[np.nonzero(score)])
        score[score == 0] = np.random.uniform(0, min_value, size=score[score == 0].shape)

        return score

    score1 = get_view_score(prefix, no_num, alpha=1)  # global view
    score2 = get_view_score(prefix, no_num, alpha=0.25)  # primary_view
    score3 = get_view_score(prefix, no_num, alpha=1.1)  # longest view
    scores = np.stack((score1, score2, score3), axis=1)

    return scores



def ted_score_multiview(holistic_views, primary_views, longest_view, no_num=False):
    def holistic_view_score(holistic_views, no_num):
        expr = []
        for d in holistic_views:
            expr.append(' '.join(d))
        score = np.zeros((len(expr), len(expr)))
        for i in range(len(expr)):
            for j in range(i, len(expr)):
                tree1 = from_postfix_to_tree(expr[i].split(' '), no_num)
                tree2 = from_postfix_to_tree(expr[j].split(' '), no_num)
                len1 = len(expr[i].split(' '))
                len2 = len(expr[j].split(' '))
                tree_dis = 1 - simple_distance(tree1, tree2) / (len1 + len2)
                score[i, j] = tree_dis
                score[j, i] = tree_dis
        return score

    def primary_view_score(primary_views):
        expr = []
        for d in primary_views:
            expr.append(d)  # root node and it's left child node
        score = np.zeros((len(expr), len(expr)))
        for i in range(len(expr)):
            for j in range(i, len(expr)):
                tree_dis = 1 - path_sim(expr[i], expr[j]) / (len(expr[i]) + len(expr[j]))
                score[i, j] = tree_dis
                score[j, i] = tree_dis
        return score

    def longest_view_score(longest_view):
        expr = []
        for d in longest_view:
            expr.append(d)
        score = np.zeros((len(expr), len(expr)))
        for i in range(len(expr)):
            for j in range(i, len(expr)):
                tree_dis = 1 - path_sim(expr[i], expr[j]) / (len(expr[i]) + len(expr[j]))
                score[i, j] = tree_dis
                score[j, i] = tree_dis
        return score

    def path_sim(path1, path2):
        length = len(path1) if len(path1) <= len(path2) else len(path2)
        diff_node = 0
        for i in range(length):
            if path1[i] != path2[i]:
                diff_node += 1
        diff_node += abs(len(path1) - len(path2))
        return diff_node

    score1 = holistic_view_score(holistic_views, no_num)
    score2 = primary_view_score(primary_views)
    score3 = longest_view_score(longest_view)
    scores = np.stack((score1, score2, score3), axis=1)
    return scores

class Solver(nn.Module):
    def __init__(self, encoder, decoder1, Subspace):
        super().__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.Subspace = Subspace

    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        torch.save(self.decoder1.state_dict(), save_directory + "/decoder1.pt")
        torch.save(self.Subspace.state_dict(), save_directory + "/Subspace.pt")


class Subspace(nn.Module):
    """
    Subspace.
    """
    def __init__(self, hidden_dim, subspace_dim, len_subspace):  # 768, 128, 3
        super(Subspace, self).__init__()

        # self.trigger_subspaces = trigger_subspaces.split(',') if trigger_subspaces else self.subspaces
        # self.trigger_indices = [self.subspaces.index(subspace) for subspace in self.trigger_subspaces]
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.len_subspace = len_subspace
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.subspace_dim * len_subspace),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.projection(x)  #x:[16, 768]  out:[16, 384]
        out = out.reshape(x.size(0), self.len_subspace, self.subspace_dim)
        # out = torch.cat([out[:, index: index + 1, :] for index in self.trigger_indices], dim=1)
        out = F.normalize(out, dim=-1, p=2)
        return out  # [16, 3, 128]


def masked_cross_entropy(logits, target, mask):
    target[~mask] = 0
    logits_flat = logits.reshape(-1, logits.size(-1))
    target_flat = target.reshape(-1, 1)
    losses_flat = torch.gather(logits_flat, index=target_flat, dim=1)
    losses = losses_flat.reshape(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / logits.size(0)
    return loss


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def cl_simpler(features, score):
    criterion = SupConLoss()
    loss = criterion(features, mask=torch.from_numpy(score).to(features.device))

    return loss


def cl_simclr(features, temperature=0.07):
    batch_size = features.size(0)
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.matmul(features, features.T)
    labels = torch.arange(batch_size).cuda()
    logits = sim_matrix / temperature
    loss = F.cross_entropy(logits, labels)

    return loss



def train_tree(args, solver, encoded, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads,
               op_tokens, constant_tokens, postfix, prefix, root_nodes, longest_view):
    encoder_outputs = encoded['text'].transpose(0, 1)
    problem_output = encoder_outputs[0]
    all_nums_encoder_outputs = encoded['num']
    target = equ_ids.clone()
    # target_neg = equ_ids_neg.clone()
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    batch_size, max_target_length = equ_ids.shape
    all_node_outputs = []
    num_start = len(op_tokens)
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    padding_hidden = torch.zeros(1, solver.decoder1.predict.hidden_size, dtype=encoder_outputs.dtype,
                                 device=encoder_outputs.device)
    constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=encoder_outputs.dtype,
                               device=encoder_outputs.device)
    operand_pads = torch.cat((constant_pads, num_pads), dim=1)
    q_list = []
    # filter the q_list by target
    # print(max_target_length)
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = solver.decoder1.predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, 1 - text_pads,
                                                                                                 1 - operand_pads)
        if t < max_target_length - 1:
            pos_op = target[:, t + 1] < 5
            q_list.append(current_embeddings[pos_op, :, :])

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        generate_input = equ_ids[:, t].clone()
        generate_input[generate_input >= len(op_tokens)] = 0
        left_child, right_child, node_label = solver.decoder1.generate(current_embeddings, generate_input,
                                                                       current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, equ_ids[:, t].contiguous().tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = solver.decoder1.merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    all_node_outputs = all_node_outputs.to(target.device)
    all_node_outputs = -torch.log_softmax(all_node_outputs, dim=-1)

    loss1 = masked_cross_entropy(all_node_outputs, target, equ_pads.bool())

    features = solver.Subspace(problem_output)

    loss_cl = 0
    if args.CL == 'SimplerCL':
        if args.similarity == 'TLWD':
            scores = tlwd_score_multiview(prefix)  # TLWD Metric
        elif args.similarity == 'TED':
            scores = ted_score_multiview(prefix, root_nodes, longest_view)  # TED Metric

        if args.H:
            loss_c1 = cl_simpler(torch.unsqueeze(features[:, 0, :], 1), scores[:, 0, :])
        else:
            loss_c1 = 0
        if args.P:
            loss_c2 = cl_simpler(torch.unsqueeze(features[:, 0, :], 1), scores[:, 1, :])
        else:
            loss_c2 = 0
        if args.L:
            loss_c3 = cl_simpler(torch.unsqueeze(features[:, 0, :], 1), scores[:, 2, :])
        else:
            loss_c3 = 0
        loss_cl = loss_c1 + loss_c2 + loss_c3
    elif args.CL == 'SimCLR':
        loss_cl = cl_simclr(problem_output)
    elif args.CL == 'NoCL':
        loss_cl = loss_cl

    return loss1, loss_cl


def gather_logits_labels(logits, labels, equ_pads):
    mask = (equ_pads).float()
    new_logits = logits.clone()  # Create a copy to avoid in-place modification
    output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = output * mask  # B * L
    return output


def get_score(logit_label, labels, equ_pads):
    mask = (equ_pads).float()
    length = mask.sum(-1)
    scores = logit_label.sum(-1) / (length)
    # print(scores.shape)
    return scores


def rrhf_loss(scores, rw_scores):
    # print(scores)
    # print(rw_scores)
    diff = (scores - rw_scores)
    diff_pos = diff > 0
    # print(diff[diff_pos])
    return diff[diff_pos].sum()


def ranking_loss_compute(all_node_outputs, target, target_neg, equ_pads, equ_pads_neg):
    logit_label_target = gather_logits_labels(all_node_outputs, target, equ_pads.bool())
    score_target = get_score(logit_label_target, target, equ_pads.bool())
    logit_label_target_neg = gather_logits_labels(all_node_outputs, target_neg, equ_pads_neg.bool())
    score_neg = get_score(logit_label_target_neg, target_neg, equ_pads_neg.bool())
    loss = rrhf_loss(score_target, score_neg)
    return loss


def evaluate_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length,
                  beam_size=3):
    encoder_outputs = encoded['text'].transpose(0, 1)
    problem_output = encoder_outputs[0]
    all_nums_encoder_outputs = encoded['num']
    batch_size = text_ids.size(0)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    num_start = len(op_tokens)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    padding_hidden = torch.zeros(1, solver.decoder1.predict.hidden_size, dtype=encoder_outputs.dtype,
                                 device=encoder_outputs.device)
    constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=num_pads.dtype, device=encoder_outputs.device)
    operand_pads = torch.cat((constant_pads, num_pads), dim=1)

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = solver.decoder1.predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, 1 - text_pads,
                                                                                                      1 - operand_pads)

            out_score = torch.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    generate_input = generate_input.to(encoder_outputs.device)
                    left_child, right_child, node_label = solver.decoder1.generate(current_embeddings, generate_input,
                                                                                   current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = solver.decoder1.merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]


def train_double(args, solver, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads,
                 op_tokens, constant_tokens, postfix, prefix, root_nodes, longest_view):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)

    loss, loss_cl = train_tree(args, solver, encoded, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads,
                               op_tokens, constant_tokens, postfix, prefix, root_nodes, longest_view)
    return loss, loss_cl


def evaluate_double(solver, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length,
                    beam_size=3):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    tree_res = evaluate_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens,
                             max_length, beam_size)
    return tree_res
