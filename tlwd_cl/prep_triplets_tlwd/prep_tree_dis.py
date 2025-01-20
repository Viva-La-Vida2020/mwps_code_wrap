import copy
import json
import torch
import random
import numpy as np
from zss import simple_distance, Node
import pandas as pd
from build_tree import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('seed:', seed)


def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# def load_data_csv(filename):
#     data = []
#     df = pd.read_csv('../data/SVAMP.csv')
#     for i, row in df.iterrows():
#         data.append(row.to_dict())
#     return data


def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        else:
            return None
    if len(st) == 1:
        return st.pop()



def from_prefix_to_postfix(prefix):
    # Stack for storing operands
    stack = []
    operators = set(['+', '-', '*', '/', '^'])
    # Reversing the order
    prefix = prefix[::-1]
    # iterating through individual tokens
    for i in prefix:
        # if token is operator
        if i in operators:
            # pop 2 elements from stack
            a = stack.pop()
            b = stack.pop()
            # concatenate them as operand1 +
            # operand2 + operator
            temp = a + ' ' + b + ' ' + i
            stack.append(temp)
        # else if operand
        else:
            stack.append(i)

    # printing final output
    return stack.pop().split()


def from_postfix_to_tree(postfix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
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


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def tree_dis_max(root, alpha):
    if root:
        current_dis = 1
        sub_tree_dis = tree_dis_recursive(root.left, alpha) + tree_dis_recursive(root.right, alpha)
        return current_dis + alpha * sub_tree_dis
    else:
        return 0


def preprocess_treedis_tlwd(data_path, save_path, alpha):
    set_seed()
    '读取数据，以list[dict{}]存储'
    train_data = load_data(data_path)
    'SVAMP中expr以prefix存储，需要转成postfix'

    expr = set()
    for d in train_data:
        expr.add(' '.join(d['postfix_normed']))

    expr = list(expr)
    res = dict()
    for i in range(len(expr)):
        for j in range(i, len(expr)):
            infix1 = from_postfix_to_infix(expr[i].split())
            prefix1 = from_infix_to_prefix(infix1.split())
            tree1 = build_tree(prefix1)
            subtree_norm(tree1)

            infix2 = from_postfix_to_infix(expr[j].split())
            prefix2 = from_infix_to_prefix(infix2.split())
            tree2 = build_tree(prefix2)
            subtree_norm(tree2)

            combined_tree = combine_trees(tree1, tree2)
            tree_dis = tree_dis_recursive(combined_tree, alpha=alpha)

            res[expr[i] + ' ; ' + expr[j]] = tree_dis
            res[expr[j] + ' ; ' + expr[i]] = tree_dis
        if i % 10 == 0:
            print(i/len(expr))

    f = open(save_path, 'w', encoding='utf-8')
    for d in res.items():
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()



def preprocess_treedis_ted(data_path, save_path):
    set_seed()
    '读取数据，以list[dict{}]存储'
    train_data = load_data(data_path)
    'SVAMP中expr以prefix存储，需要转成postfix'

    expr = set()
    for d in train_data:
        expr.add(' '.join(d['postfix_normed']))

    expr = list(expr)
    res = dict()
    for i in range(len(expr)):
        for j in range(i, len(expr)):
            tree1 = from_postfix_to_tree(expr[i].split(' '))
            tree2 = from_postfix_to_tree(expr[j].split(' '))
            tree_dis = simple_distance(tree1, tree2)

            res[expr[i] + ' ; ' + expr[j]] = tree_dis
            res[expr[j] + ' ; ' + expr[i]] = tree_dis
        if i % 10 == 0:
            print(i/len(expr))

    f = open(save_path, 'w', encoding='utf-8')
    for d in res.items():
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()



# if __name__ == '__main__':
#     data_root_path = '../data/math23k_RecursiveTreeDis/'
#     preprocess_treedis(data_root_path)