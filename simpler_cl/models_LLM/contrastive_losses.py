import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from zss import simple_distance, Node


class Subspace(nn.Module):
    def __init__(self, hidden_dim, subspace_dim, len_subspace):  # 768, 128, 3
        super(Subspace, self).__init__()
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.len_subspace = len_subspace
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.subspace_dim * len_subspace),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.projection(x)  #x:(N,H=576)  out:(N,H=384)
        out = out.reshape(x.size(0), self.len_subspace, self.subspace_dim)
        out = F.normalize(out, dim=-1, p=2)
        return out  # [N, 3, 128]


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.subtree_length = 1


class FatherTreeNode:
    def __init__(self, value_A, value_B):
        self.valueA = value_A
        self.valueB = value_B
        self.left = None
        self.right = None


def TLWD_score_multiview(prefix, no_num):
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


    score1 = get_view_score(prefix, no_num, alpha=1)
    score2 = get_view_score(prefix, no_num, alpha=0.25)
    score3 = get_view_score(prefix, no_num, alpha=1.1)
    scores = np.stack((score1, score2, score3), axis=1)

    return scores


def calculate_tlwd(prefix1, prefix2, alpha, VU=False):
    if VU:
        ops = ['+', '-', '*', '/', '^']
        prefix1_normed = [op if op in ops else 'N' for op in prefix1.split()]
        prefix2_normed = [op if op in ops else 'N' for op in prefix2.split()]
        root_A = build_tree(prefix1_normed)
        root_B = build_tree(prefix2_normed)

        subtree_norm(root_A)
        subtree_norm(root_B)

        combined_root = combine_trees(root_A, root_B)
        tree_dis = tree_dis_recursive_va(combined_root, alpha)
        dis_max = tree_dis_max_va(combined_root, alpha)
    else:
        root_A = build_tree(prefix1.split())
        root_B = build_tree(prefix2.split())

        subtree_norm(root_A)
        subtree_norm(root_B)

        combined_root = combine_trees(root_A, root_B)
        tree_dis = tree_dis_recursive(combined_root, alpha)
        dis_max = tree_dis_max(combined_root, alpha)
    sim = 1 - tree_dis / dis_max

    return sim

def subtree_norm(root):
    if root:
        if root.value == '+' or root.value == '*':
            if root.left.subtree_length < root.right.subtree_length:
                tem = copy.deepcopy(root.left)
                root.left = root.right
                root.right = tem
        subtree_norm(root.left)
        subtree_norm(root.right)
    else:
        return


def combine_trees(root_A, root_B):
    if root_A is None and root_B is None:
        return None

    combined_root = FatherTreeNode(
        value_A=root_A.value if root_A else None,
        value_B=root_B.value if root_B else None
    )

    combined_root.left = combine_trees(root_A.left if root_A else None, root_B.left if root_B else None)
    combined_root.right = combine_trees(root_A.right if root_A else None, root_B.right if root_B else None)

    return combined_root
def build_tree(prefix_expression):
    if not prefix_expression:
        return None

    stack = []

    for token in reversed(prefix_expression):
        if 'N' in token or 'C' in token:
            stack.append(TreeNode(token))
        else:
            operand1 = stack.pop()
            operand2 = stack.pop()
            node = TreeNode(token)
            node.left = operand1
            node.right = operand2
            node.subtree_length = 1 + operand1.subtree_length + operand2.subtree_length
            stack.append(node)

    return stack[0]


def tree_dis_recursive_va(root, alpha):
    if root:
        ops = ['+', '-', '*', '/', '^']
        placeholders = [None, 'N']
        if (root.valueA in ops and root.valueB in ops) and (root.valueA == root.valueB):
            current_dis = 0
        elif root.valueA in placeholders and root.valueB in placeholders:
            current_dis = 0
        else:
            current_dis = 1
        sub_tree_dis = tree_dis_recursive(root.left, alpha) + tree_dis_recursive(root.right, alpha)
        return current_dis + alpha * sub_tree_dis
    else:
        return 0

def tree_dis_max(root, alpha):
    if root:
        current_dis = 1
        sub_tree_dis = tree_dis_recursive(root.left, alpha) + tree_dis_recursive(root.right, alpha)
        return current_dis + alpha * sub_tree_dis
    else:
        return 0

def tree_dis_max_va(root, alpha):
    if root:
        current_dis = 1
        sub_tree_dis = tree_dis_recursive_va(root.left, alpha) + tree_dis_recursive_va(root.right, alpha)
        return current_dis + alpha * sub_tree_dis
    else:
        return 0


def tree_dis_recursive(root, alpha):
    if root:
        if root.valueA == root.valueB:
            current_dis = 0
        else:
            current_dis = 1
        sub_tree_dis = tree_dis_recursive(root.left, alpha) + tree_dis_recursive(root.right, alpha)
        return current_dis + alpha * sub_tree_dis
    else:
        return 0


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


class ContraCLMSeqLoss(nn.Module):
    def __init__(self, pad_token_id, temperature=0.05):
        super(ContraCLMSeqLoss, self).__init__()
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        print(f"Sequence-Level Contrastive Loss:\t temperature: {temperature}")

    def forward(self, last_hidden_states_1, last_hidden_states_2, token_mask):
        device = last_hidden_states_1.device  # [N, L, H]
        batch_size = last_hidden_states_1.size(0)

        # get the sequence representation via mean pooling
        token_mask = token_mask.unsqueeze(-1)  # [N, L, 1]
        features_1 = torch.sum(last_hidden_states_1 * token_mask, dim=1) / torch.sum(token_mask, dim=1)  # [N, H]
        features_2 = torch.sum(last_hidden_states_2 * token_mask, dim=1) / torch.sum(token_mask, dim=1)  # [N, H]
        features_1, features_2 = F.normalize(features_1, dim=1), F.normalize(features_2, dim=1)  # [N, H]
        features = torch.cat([features_1, features_2], dim=0)  # [2N, H]

        # create block diagonal mask to avoid contrast within the neighborhood of each example
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1).to(torch.float32) / self.temperature)  # [N]
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()).to(torch.float32) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (Ng + pos))).mean()

        return loss