import numpy as np
from zss import simple_distance, Node


def from_prefix_to_postfix(prefix):
    stack = []
    operators = {'+', '-', '*', '/', '^'}

    for i in reversed(prefix):
        if i in operators:
            a = stack.pop()
            b = stack.pop()
            temp = a + ' ' + b + ' ' + i
            stack.append(temp)
        else:
            stack.append(i)

    return stack.pop().split()


def from_postfix_to_tree(postfix):
    stack = []
    operators = {"+", "-", "^", "*", "/"}

    for p in postfix:
        if p not in operators:
            stack.append(Node(p))
        else:
            if len(stack) < 2:
                raise ValueError(f"Invalid postfix expression: {postfix}")
            right, left = stack.pop(), stack.pop()
            stack.append(Node(p).addkid(left).addkid(right))

    return stack.pop() if stack else None


def path_sim(path1, path2):
    diff_node = sum(x != y for x, y in zip(path1, path2))
    diff_node += abs(len(path1) - len(path2))
    return diff_node


def holistic_view_score(holistic_views):
    expr = [' '.join(d) for d in holistic_views]
    size = len(expr)
    score = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            tree1 = from_postfix_to_tree(from_prefix_to_postfix(expr[i].split()))
            tree2 = from_postfix_to_tree(from_prefix_to_postfix(expr[j].split()))
            len1, len2 = len(expr[i].split()), len(expr[j].split())
            tree_dis = 1 - simple_distance(tree1, tree2) / (len1 + len2)
            score[i, j] = score[j, i] = tree_dis

    return score


def view_score(views):
    """primary view/ longest view"""
    expr = [' '.join(d) for d in views]
    size = len(expr)
    score = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            tree_dis = 1 - path_sim(expr[i], expr[j]) / (len(expr[i]) + len(expr[j]))
            score[i, j] = score[j, i] = tree_dis

    return score


def multiview_score_ted(holistic_views, primary_views, longest_view):
    score1 = holistic_view_score(holistic_views)
    score2 = view_score(primary_views)
    score3 = view_score(longest_view)
    return np.stack((score1, score2, score3), axis=1)