import json


def load_json(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data, filename):
    f = open(filename, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def build_tree(sequence):
    def build_helper(index):
        if index >= len(sequence):
            return None, index

        if 'N' in sequence[index] or 'C' in sequence[index]:
            return TreeNode(sequence[index]), index + 1
        else:
            root = TreeNode(sequence[index])
            root.left, index = build_helper(index + 1)
            root.right, index = build_helper(index)
            return root, index

    root, _ = build_helper(0)
    return root


def print_preorder(root):
    if root:
        print(root.value, end=" ")
        print_preorder(root.left)
        print_preorder(root.right)


def find_longest_path(root):
    if not root:
        return []

    def dfs(node):
        if not node:
            return []
        if not node.left and not node.right:
            return [[node.value]]
        left_paths = dfs(node.left)
        right_paths = dfs(node.right)
        all_paths = left_paths + right_paths
        for path in all_paths:
            path.insert(0, node.value)
        return all_paths

    paths = dfs(root)
    longest_path = max(paths, key=len)
    return longest_path

def find_root_nodes(root):
    if not root:
        return []
    return [root.value, root.left.value, root.right.value] # -, N, +;  -, +, N


data_path = "../../data/mathqa/"

train = load_json(data_path + "train_cl.jsonl")
for d in train:
    if len(d['prefix']) == 1:
        root_nodes = d['prefix']
        longest_path = d['prefix']
    else:
        root = build_tree(d['prefix'])
        longest_path = find_longest_path(root)
    d['longest_view'] = longest_path
    root_nodes = find_root_nodes(root)
    ops = '+ - * / ^'.split()
    # root_nodes_N = [op if op in ops else 'N' for op in root_nodes]
    # print(longest_path[:-1], root_nodes_N)
    d['root_nodes'] = root_nodes
save_json(train, data_path + "train.jsonl")