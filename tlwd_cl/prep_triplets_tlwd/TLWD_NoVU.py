import copy


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


def build_tree(prefix_expression):
    if not prefix_expression:
        return None

    stack = []

    for token in reversed(prefix_expression):
        if 'N' in token:
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


def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(f"{root.value}", end=' ')
        # print(f"{root.value} (Subtree Length: {root.subtree_length})", end=' ')
        inorder_traversal(root.right)


def preorder_traversal(root):
    if root:
        current_prefix = [root.value]
        current_prefix.extend(preorder_traversal(root.left))
        current_prefix.extend(preorder_traversal(root.right))
        return current_prefix
    else:
        return []


def subtree_norm(root):
    if root:
        if root.value == '+' or root.value == '*':
            if root.left.subtree_length < root.right.subtree_length:  # 左子树长度小于右子树
                tem = copy.deepcopy(root.left)
                root.left = root.right  # 交换左右子树
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


def inorder_traversal_combined(root):
    if root:
        inorder_traversal_combined(root.left)
        print(f"ValueA: {root.valueA}, ValueB: {root.valueB}")
        inorder_traversal_combined(root.right)


def preorder_traversal_combined(root):
    if root:
        print(f"ValueA: {root.valueA}, ValueB: {root.valueB}")
        preorder_traversal_combined(root.left)
        preorder_traversal_combined(root.right)


def tree_dis_recursive(root, alpha):
    if root:
        ops = ['+', '-', '*', '/', '^']
        # placeholders = [None, 'N']
        # if (root.valueA in ops and root.valueB in ops) and (root.valueA == root.valueB):
        #     current_dis = 0
        # elif root.valueA in placeholders and root.valueB in placeholders:
        #     current_dis = 0
        # else:
        #     current_dis = 1
        if root.valueA == root.valueB:  # 两个结点相同
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

# 示例用法
if __name__ == "__main__":
    '根据prefix构建recursive tree'
    "N N - N * N + N - ; N N * N / N /"
    root_A = build_tree("+ + - N_0 N_1 N_2 + N_3 - N_4 N_5".split())
    root_B = build_tree("+ + + N_0 N_1 N_2 + N_3 - N_4 N_6".split())
    prefix = preorder_traversal(root_A)
    print(prefix)

    '根据交换律对recursive tree进行正则'
    subtree_norm(root_A)
    subtree_norm(root_B)
    prefix_normed = preorder_traversal(root_A)
    print(prefix_normed)

    '找到两棵树的最小公共父树'
    combined_root = combine_trees(root_A, root_B)
    print("Inorder Traversal of Combined Tree:")
    preorder_traversal_combined(combined_root)

    '计算Recursive Tree Distance'
    tree_dis = tree_dis_recursive(combined_root, alpha=0.5)
    print('Tree Distance: ', tree_dis)

    '计算当前两棵树的理论最大Tree Distance'
    tree_dis_max = tree_dis_max(combined_root, alpha=0.5)
    print('Max Tree Distance: ', tree_dis_max)

    '计算similarity'
    sim = 1 - tree_dis / tree_dis_max
    print('Similarity: ', sim)