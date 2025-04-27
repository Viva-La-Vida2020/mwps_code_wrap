"""
using TED (tree edit distance) for logic similarity calculation used
"""

import numpy as np
from zss import simple_distance, Node


def from_prefix_to_postfix(prefix):
    """
    Convert an expression from prefix notation to postfix notation.

    Args:
        prefix (list[str]): The expression in prefix order.

    Returns:
        list[str]: The expression in postfix order.
    """
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
    """
    Build an expression tree from a postfix expression.

    Args:
        postfix (list[str]): The expression in postfix order.

    Returns:
        Node: The root node of the expression tree.
    """
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
    """
    Compute the difference between two paths based on node mismatch and length difference.

    Args:
        path1 (list[str]): First path.
        path2 (list[str]): Second path.

    Returns:
        int: The number of different nodes plus length difference.
    """
    diff_node = sum(x != y for x, y in zip(path1, path2))
    diff_node += abs(len(path1) - len(path2))
    return diff_node


def holistic_view_score(holistic_views):
    """
    Compute pairwise similarity scores between holistic views using tree edit distance.

    Args:
        holistic_views (list[list[str]]): List of holistic view expressions.

    Returns:
        np.ndarray: A (size, size) matrix of similarity scores between views.
    """
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
    """
    Compute pairwise similarity scores between primary or longest views based on path similarity.

    Args:
        views (list[list[str]]): List of views (as lists of tokens).

    Returns:
        np.ndarray: A (size, size) matrix of similarity scores between views.
    """
    expr = [' '.join(d) for d in views]
    size = len(expr)
    score = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            tree_dis = 1 - path_sim(expr[i], expr[j]) / (len(expr[i]) + len(expr[j]))
            score[i, j] = score[j, i] = tree_dis

    return score


def multiview_score_ted(holistic_views, primary_views, longest_view):
    """
    Compute multi-view similarity scores using tree edit distance and path similarity.

    Args:
        holistic_views (list[list[str]]): Holistic views (full expression views).
        primary_views (list[list[str]]): Primary views (main reasoning paths).
        longest_view (list[list[str]]): Longest views (longest paths extracted).

    Returns:
        np.ndarray: A (size, 3) matrix stacking three different similarity scores.
    """
    score1 = holistic_view_score(holistic_views)
    score2 = view_score(primary_views)
    score3 = view_score(longest_view)
    return np.stack((score1, score2, score3), axis=1)
