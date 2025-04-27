"""
Tree structure definition, segmentation based on different views and similarity calculation from different views.
This similarity will be used in Similarity weighted Contrastive loss.
"""

import copy
import numpy as np


class TreeNode:
    """
    Contruction of a Tree
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.subtree_length = 1

    @staticmethod
    def build_tree(preifx):
        """
        build the tree according prefix expression
        """
        if not preifx:
            return None

        stack = []
        for token in reversed(preifx):
            if 'N' in token or 'C' in token:
                stack.append(TreeNode(token))
            else:
                if len(stack) < 2:
                    raise ValueError(f"Invalid prefix expression: {preifx}")
                operand1, operand2 = stack.pop(), stack.pop()
                node = TreeNode(token)
                node.left, node.right = operand1, operand2
                node.subtree_length = 1 + operand1.subtree_length + operand2.subtree_length
                stack.append(node)

        return stack[0] if stack else None

    @staticmethod
    def normalize_subtree(root):
        """
        tree Normalization
        """
        if root:
            if root.value in ['+', '*'] and root.left.subtree_length < root.right.subtree_length:
                root.left, root.right = root.right, copy.deepcopy(root.left)
        else:
            return
        if root.left:
            TreeNode.normalize_subtree(root.left)
        if root.right:
            TreeNode.normalize_subtree(root.right)


class OverlapTreeNode:
    """
    overlap of two trees
    """
    def __init__(self, value_a, value_b):
        self.value_a = value_a
        self.value_b = value_b
        self.left = None
        self.right = None

    @staticmethod
    def combine_trees(root_a, root_b):
        """
        recursive calling the overlap over the node to combine the whole tree
        """
        if not root_a and not root_b:
            return None
        combined_root = OverlapTreeNode(
            value_a=root_a.value if root_a else None,
            value_b=root_b.value if root_b else None
        )
        combined_root.left = OverlapTreeNode.combine_trees(
            root_a.left if root_a else None, root_b.left if root_b else None
        )
        combined_root.right = OverlapTreeNode.combine_trees(
            root_a.right if root_a else None, root_b.right if root_b else None
        )
        return combined_root


def tree_distance(root, alpha, variable_unification=False):
    """
    distance between two trees
    """
    if not root:
        return 0

    ops = {'+', '-', '*', '/', '^'}
    placeholders = {None, 'N'}

    if variable_unification:
        if root.value_a in ops and root.value_b in ops and root.value_a == root.value_b:
            current_dis = 0
        elif root.value_a in placeholders and root.value_b in placeholders:
            current_dis = 0
        else:
            current_dis = 1
    else:
        current_dis = 0 if root.value_a == root.value_b else 1

    sub_tree_dis = (tree_distance(root.left, alpha, variable_unification) +
                    tree_distance(root.right, alpha, variable_unification))
    return current_dis + alpha * sub_tree_dis


def tree_distance_max(root, alpha, variable_unification=False):
    """
    maximum of the distance between two trees
    """
    if not root:
        return 0
    return 1 + alpha * (tree_distance_max(root.left, alpha, variable_unification) +
                        tree_distance_max(root.right, alpha, variable_unification))


def calculate_tlwd(prefix1, prefix2, alpha, variable_unification=False):
    """
    tlwd similarity calculation: please refer to
    On the Selection of Positive and Negative Samples for Contrastive Math Word Problem Neural Solver
    Y Li, L Wang, JJ Kim, CS Tan, Y Luo
    Proceedings of the 17th International Conference on Educational Data Mining
    """
    ops = {'+', '-', '*', '/', '^'}

    if variable_unification:
        prefix1 = [op if op in ops else 'N' for op in prefix1.split()]
        prefix2 = [op if op in ops else 'N' for op in prefix2.split()]
    else:
        prefix1 = prefix1.split()
        prefix2 = prefix2.split()

    root_a, root_b = TreeNode.build_tree(prefix1), TreeNode.build_tree(prefix2)

    TreeNode.normalize_subtree(root_a)
    TreeNode.normalize_subtree(root_b)

    combined_root = OverlapTreeNode.combine_trees(root_a, root_b)
    tree_dis = tree_distance(combined_root, alpha, variable_unification)
    dis_max = tree_distance_max(combined_root, alpha, variable_unification)

    return 1 - tree_dis / dis_max


def multiview_score_tlwd(prefix, variable_unification=False):
    """
    three different views similarity using tlwd:hoslitstic,longest,primary
    please refer to our "Simpler" paper
    """
    def get_view_score(prefix, variable_unification, alpha):
        expr = [' '.join(d) for d in prefix]
        size = len(expr)
        score = np.zeros((size, size))

        for i in range(size):
            for j in range(i, size):
                sim_tlwd = calculate_tlwd(expr[i], expr[j], alpha, variable_unification)
                score[i, j] = score[j, i] = sim_tlwd

        min_value = np.min(score[np.nonzero(score)])
        score[score == 0] = np.random.uniform(0, min_value, size=score[score == 0].shape)

        return score

    score1 = get_view_score(prefix, variable_unification, alpha=1.0)  # Global view
    score2 = get_view_score(prefix, variable_unification, alpha=0.25)  # Primary view
    score3 = get_view_score(prefix, variable_unification, alpha=1.1)  # Longest view

    return np.stack((score1, score2, score3), axis=1)