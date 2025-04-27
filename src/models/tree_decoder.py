"""
Decoder using tree structure: model initalization and saving, shared by all encoder and tree decoder models\
This is used and adapted from: GTS in https://github.com/arkilpatel/SVAMP
"""

import copy
import torch
from torch import nn


class TreeNode:
    """
    TreeNode represents a node in a tree structure.

    Args:
        embedding (Tensor): The embedding vector associated with the node.
        left_flag (bool, optional): Indicator if the node is a left child. Defaults to False.
    """
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


def copy_list(l):
    """
    Deep copies a list, recursively copying nested lists.

    Args:
        l (list): The list to copy.

    Returns:
        list: A deep copied version of the input list.
    """
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:
    """
    TreeBeam represents a beam search node used during decoding.

    Args:
        score (float): The score associated with the beam.
        node_stack (list): Stack of tree nodes.
        embedding_stack (list): Stack of embeddings.
        left_childs (list): List of left child nodes.
        out (object): Output associated with the beam.
    """
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:
    """
    TreeEmbedding represents an embedding within the tree structure.

    Args:
        embedding (Tensor): Embedding vector.
        terminal (bool, optional): Whether the node is a terminal node. Defaults to False.
    """
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class Score(nn.Module):
    """
    Score computes attention scores between hidden states and candidate embeddings.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden features.
    """
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        """
        Compute the score for each candidate embedding.

        Args:
            hidden (Tensor): Hidden state tensor of shape (B, 1, H).
            num_embeddings (Tensor): Candidate embeddings tensor of shape (B, O, H).
            num_mask (Tensor, optional): Mask for invalid candidates. Defaults to None.

        Returns:
            Tensor: Computed scores of shape (B, O).
        """
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)

        return score


class TreeAttn(nn.Module):
    """
    TreeAttn implements a tree-specific attention mechanism.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden features.
    """
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        """
        Apply attention mechanism over encoder outputs.

        Args:
            hidden (Tensor): Current hidden state (1, B, H).
            encoder_outputs (Tensor): Encoder outputs (S, B, H).
            seq_mask (Tensor, optional): Mask for padding tokens. Defaults to None.

        Returns:
            Tensor: Attention weights (B, 1, S).
        """
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class Prediction(nn.Module):
    """
    Prediction is a seq2tree decoder with problem-aware dynamic encoding.

    Args:
        hidden_size (int): Size of the hidden features.
        op_nums (int): Number of operation symbols.
        input_size (int): Size of the input embeddings.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums
        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))
        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        """
        Perform one decoding step: predict operator scores and number scores.

        Args:
            node_stacks (List[List[TreeNode]]): Current node stacks for each example in batch.
            left_childs (List[torch.Tensor]): Left child node embeddings.
            encoder_outputs (torch.Tensor): Encoder hidden states. (seq_len, batch_size, hidden_size)
            num_pades (torch.Tensor): Dynamic encoding of number tokens.
            padding_hidden (torch.Tensor): Hidden state used for padding empty stacks.
            seq_mask (torch.Tensor): Mask for valid input tokens. (batch_size, seq_len)
            mask_nums (torch.Tensor): Mask for valid number tokens. (batch_size, number_size)

        Returns:
            num_score (torch.Tensor): Scores over number candidates. (batch_size, number_size)
            op (torch.Tensor): Scores over operator candidates. (batch_size, op_nums)
            current_node (torch.Tensor): Updated current node hidden state. (batch_size, hidden_size)
            current_context (torch.Tensor): Context vector from attention over encoder outputs.
            (batch_size, 1, hidden_size)
            embedding_weight (torch.Tensor): Expanded embedding weights for numbers.
            (batch_size, input_size + number_size, hidden_size)
        """
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    """Generate left and right child node embeddings for tree decoding."""

    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        """
        Args:
            hidden_size (int): Hidden state dimension.
            op_nums (int): Number of operator tokens.
            embedding_size (int): Dimension of operator embeddings.
            dropout (float): Dropout probability.
        """
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        """
        Generate left and right child embeddings given parent node and context.

        Args:
            node_embedding (torch.Tensor): Current node hidden representation. (batch_size, 1, hidden_size)
            node_label (torch.Tensor): Label indices for current node. (batch_size,)
            current_context (torch.Tensor): Context vector from encoder. (batch_size, 1, hidden_size)

        Returns:
            l_child (torch.Tensor): Left child node embedding. (batch_size, hidden_size)
            r_child (torch.Tensor): Right child node embedding. (batch_size, hidden_size)
            node_label_emb (torch.Tensor): Embedded node label. (batch_size, embedding_size)
        """
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    """Merge two subtree embeddings into one parent node embedding."""

    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        """
        Args:
            hidden_size (int): Hidden state dimension.
            embedding_size (int): Dimension of node embeddings.
            dropout (float): Dropout probability.
        """
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        """
        Merge two child nodes and parent node embedding to form a subtree.

        Args:
            node_embedding (torch.Tensor): Parent node hidden representation. (batch_size, hidden_size)
            sub_tree_1 (torch.Tensor): Left subtree embedding. (batch_size, hidden_size)
            sub_tree_2 (torch.Tensor): Right subtree embedding. (batch_size, hidden_size)

        Returns:
            sub_tree (torch.Tensor): Merged subtree embedding. (batch_size, hidden_size)
        """
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class TreeDecoder(nn.Module):
    """Decoder module for tree-structured output in math word problem solving."""

    def __init__(self, config, op_size, constant_size, embedding_size):
        """
        Args:
            config (transformers.PretrainedConfig): Transformer model configuration.
            op_size (int): Number of operator tokens.
            constant_size (int): Number of constant tokens.
            embedding_size (int): Dimension of operator embeddings.
        """
        super().__init__()
        # self.dropout = config.hidden_dropout_prob
        self.predict = Prediction(hidden_size=config.hidden_size, op_nums=op_size, input_size=constant_size)
        self.generate = GenerateNode(hidden_size=config.hidden_size, embedding_size=embedding_size, op_nums=op_size)
        self.merge = Merge(hidden_size=config.hidden_size, embedding_size=embedding_size)
