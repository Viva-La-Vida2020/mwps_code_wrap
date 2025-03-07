import copy
import numpy as np


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.subtree_length = 1

    @staticmethod
    def build_tree(preifx):
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
    def __init__(self, value_A, value_B):
        self.valueA = value_A
        self.valueB = value_B
        self.left = None
        self.right = None

    @staticmethod
    def combine_trees(root_A, root_B):
        if not root_A and not root_B:
            return None
        combined_root = OverlapTreeNode(
            value_A=root_A.value if root_A else None,
            value_B=root_B.value if root_B else None
        )
        combined_root.left = OverlapTreeNode.combine_trees(
            root_A.left if root_A else None, root_B.left if root_B else None
        )
        combined_root.right = OverlapTreeNode.combine_trees(
            root_A.right if root_A else None, root_B.right if root_B else None
        )
        return combined_root


def tree_distance(root, alpha, variable_unification=False):
    if not root:
        return 0

    ops = {'+', '-', '*', '/', '^'}
    placeholders = {None, 'N'}

    if variable_unification:
        if root.valueA in ops and root.valueB in ops and root.valueA == root.valueB:
            current_dis = 0
        elif root.valueA in placeholders and root.valueB in placeholders:
            current_dis = 0
        else:
            current_dis = 1
    else:
        current_dis = 0 if root.valueA == root.valueB else 1

    sub_tree_dis = tree_distance(root.left, alpha, variable_unification) + tree_distance(root.right, alpha, variable_unification)
    return current_dis + alpha * sub_tree_dis


def tree_distance_max(root, alpha, variable_unification=False):
    if not root:
        return 0
    return 1 + alpha * (tree_distance_max(root.left, alpha, variable_unification) + tree_distance_max(root.right, alpha, variable_unification))


def calculate_tlwd(prefix1, prefix2, alpha, variable_unification=False):
    ops = {'+', '-', '*', '/', '^'}

    if variable_unification:
        prefix1 = [op if op in ops else 'N' for op in prefix1.split()]
        prefix2 = [op if op in ops else 'N' for op in prefix2.split()]
    else:
        prefix1 = prefix1.split()
        prefix2 = prefix2.split()

    root_A, root_B = TreeNode.build_tree(prefix1), TreeNode.build_tree(prefix2)

    TreeNode.normalize_subtree(root_A)
    TreeNode.normalize_subtree(root_B)

    combined_root = OverlapTreeNode.combine_trees(root_A, root_B)
    tree_dis = tree_distance(combined_root, alpha, variable_unification)
    dis_max = tree_distance_max(combined_root, alpha, variable_unification)

    return 1 - tree_dis / dis_max


def multiview_score_tlwd(prefix, variable_unification=False):
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