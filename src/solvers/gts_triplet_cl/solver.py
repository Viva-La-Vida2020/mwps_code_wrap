"""
Solver of encoder decoder model using GTS decoder using
traditional triplet based contrastive learning.
This is used and adapted from: Textual-CL method in https://github.com/yiyunya/Textual_CL_MWP
"""

import copy
import torch
from torch import nn
from src.models.tree_decoder import TreeNode, TreeEmbedding, TreeBeam, copy_list


class Solver(nn.Module):
    """
        Solver model that combines an encoder and a tree-based decoder
        and train using traditional triplet-based CL
    """

    def __init__(self, encoder, decoder):
        """
        inital the encoder and decoder.
        Args:
            encoder (nn.Module): The encoder model that processes input sequences.
            decoder (nn.Module): The tree-based decoder model that generates output sequences.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def save_pretrained(self, save_directory):
        """
        Saves the trained encoder and decoder.
        Args:
            save_directory (str): The directory where the encoder and decoder should be saved.

        """
        self.encoder.save_pretrained(save_directory)
        torch.save(self.decoder.state_dict(), f"{save_directory}/decoder.pt")

    def train_step(self, text_ids, text_pads, num_ids, num_pads,
                   pos_text_ids, pos_text_pads, pos_num_ids, pos_num_pads,
                   neg_text_ids, neg_text_pads, neg_num_ids, neg_num_pads,
                   equ_ids, equ_pads, op_tokens, constant_tokens):
        """
        Executes a single training step.
        Args:
            text_ids (torch.Tensor): Input text token IDs for the anchor examples.
            text_pads (torch.Tensor): Padding mask for the anchor text.
            num_ids (torch.Tensor): Input numerical token IDs for the anchor examples.
            num_pads (torch.Tensor): Padding mask for the anchor numerical tokens.
            pos_text_ids (torch.Tensor): Input text token IDs for positive examples.
            pos_text_pads (torch.Tensor): Padding mask for positive text examples.
            pos_num_ids (torch.Tensor): Input numerical token IDs for positive examples.
            pos_num_pads (torch.Tensor): Padding mask for positive numerical tokens.
            neg_text_ids (torch.Tensor): Input text token IDs for negative examples.
            neg_text_pads (torch.Tensor): Padding mask for negative text examples.
            neg_num_ids (torch.Tensor): Input numerical token IDs for negative examples.
            neg_num_pads (torch.Tensor): Padding mask for negative numerical tokens.
            equ_ids (torch.Tensor): Equation token IDs (target output sequence).
            equ_pads (torch.Tensor): Padding mask for the output sequence.
            op_tokens (list): List of operator tokens used in the model.
            constant_tokens (list): List of constant tokens used in the model.

        Returns:
            tuple: A tuple containing:
                - loss_solver (torch.Tensor): The loss computed for solving the equation.
                - loss_cl (torch.Tensor): The loss computed for the contrastive learning objective.
                - encoded_anchor (dict): The encoded representations of the anchor input.
        """
        encoded_anchor = self.encoder(text_ids, text_pads, num_ids, num_pads)
        encoded_pos = self.encoder(pos_text_ids, pos_text_pads, pos_num_ids, pos_num_pads)
        encoded_neg = self.encoder(neg_text_ids, neg_text_pads, neg_num_ids, neg_num_pads)

        loss_solver = self._train_tree(encoded_anchor, text_pads, num_pads,
                                       equ_ids, equ_pads, op_tokens, constant_tokens)
        loss_cl = self._triplet_cl(encoded_anchor['text'][:, 0], encoded_pos['text'][:, 0], encoded_neg['text'][:, 0])

        return loss_solver, loss_cl, encoded_anchor

    def evaluate_step(self, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length,
                      beam_size=3):
        """
        Executes a single evaluation step.
        Args:
            text_ids (torch.Tensor): Input text token IDs for evaluation.
            text_pads (torch.Tensor): Padding mask for the text during evaluation.
            num_ids (torch.Tensor): Input numerical token IDs for evaluation.
            num_pads (torch.Tensor): Padding mask for the numerical tokens during evaluation.
            op_tokens (list): List of operator tokens used in the model.
            constant_tokens (list): List of constant tokens used in the model.
            max_length (int): Maximum length of the generated sequence during evaluation.
            beam_size (int, optional): Number of beams to use in beam search. Default is 3.

        Returns:
            TreeBeam: The best beam after evaluating the tree with beam search.
        """
        encoded = self.encoder(text_ids, text_pads, num_ids, num_pads)
        return self._evaluate_tree(encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens,
                                   max_length, beam_size)

    def _train_tree(self, encoded, text_pads, num_pads, equ_ids, equ_pads, op_tokens, constant_tokens):
        """
        Trains the tree decoder using the provided encoded representations.
        Args:
            encoded (dict): The encoded input sequences containing 'text' and 'num' keys.
            text_pads (torch.Tensor): Padding mask for the input text.
            num_pads (torch.Tensor): Padding mask for the input numerical tokens.
            equ_ids (torch.Tensor): Equation token IDs (target output sequence).
            equ_pads (torch.Tensor): Padding mask for the output sequence.
            op_tokens (list): List of operator tokens.
            constant_tokens (list): List of constant tokens.

        Returns:
            torch.Tensor: The computed loss for training the tree decoder.
        """
        encoder_outputs = encoded['text'].transpose(0, 1)
        problem_output = encoder_outputs[0]
        all_nums_encoder_outputs = encoded['num']

        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        batch_size, max_target_length = equ_ids.shape
        all_node_outputs = []
        num_start = len(op_tokens)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        padding_hidden = torch.zeros(1, self.decoder.predict.hidden_size, dtype=encoder_outputs.dtype,
                                     device=encoder_outputs.device)
        constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=encoder_outputs.dtype,
                                   device=encoder_outputs.device)
        operand_pads = torch.cat((constant_pads, num_pads), dim=1)

        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder.predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, 1 - text_pads,
                                                                                                     1 - operand_pads)

            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            generate_input = equ_ids[:, t].clone()
            generate_input[generate_input >= len(op_tokens)] = 0
            left_child, right_child, node_label = self.decoder.generate(current_embeddings, generate_input,
                                                                        current_context)

            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, equ_ids[:, t].contiguous().tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    _ = node_stack.pop()
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
                        current_num = self.decoder.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        target = equ_ids.clone()
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        all_node_outputs = all_node_outputs.to(target.device)
        all_node_outputs = -torch.log_softmax(all_node_outputs, dim=-1)
        loss = self._masked_cross_entropy(all_node_outputs, target, equ_pads.bool())
        return loss

    def _evaluate_tree(self, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length,
                       beam_size=3):
        """
        Evaluates the tree decoder by generating predictions using beam search.

        Args:
            encoded (dict): The encoded input sequences containing 'text' and 'num' keys.
            text_ids (torch.Tensor): Input text token IDs for evaluation.
            text_pads (torch.Tensor): Padding mask for the input text.
            num_ids (torch.Tensor): Input numerical token IDs for evaluation.
            num_pads (torch.Tensor): Padding mask for the numerical tokens.
            op_tokens (list): List of operator tokens.
            constant_tokens (list): List of constant tokens.
            max_length (int): Maximum length of the generated sequence during evaluation.
            beam_size (int, optional): Number of beams to use in beam search. Default is 3.

        Returns:
            TreeBeam: The best beam generated after beam search.
        """
        encoder_outputs = encoded['text'].transpose(0, 1)
        problem_output = encoder_outputs[0]
        all_nums_encoder_outputs = encoded['num']
        batch_size = text_ids.size(0)

        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        num_start = len(op_tokens)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
        padding_hidden = torch.zeros(1, self.decoder.predict.hidden_size, dtype=encoder_outputs.dtype,
                                     device=encoder_outputs.device)
        constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=num_pads.dtype,
                                   device=encoder_outputs.device)
        operand_pads = torch.cat((constant_pads, num_pads), dim=1)

        for _ in range(max_length):
            current_beams = []
            while beams:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue

                left_childs = b.left_childs
                num_score, op, current_embeddings, current_context, current_nums_embeddings = (
                    self.decoder.predict(b.node_stack, left_childs, encoder_outputs,
                                         all_nums_encoder_outputs, padding_hidden,
                                         1 - text_pads, 1 - operand_pads))

                out_score = torch.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)
                    out_token = int(ti)
                    current_out.append(out_token)

                    _ = current_node_stack[0].pop()
                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token]).to(encoder_outputs.device)
                        left_child, right_child, node_label = self.decoder.generate(current_embeddings, generate_input,
                                                                                    current_context)
                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.decoder.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))

            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)[:beam_size]
            if all(len(b.node_stack[0]) == 0 for b in beams):
                break

        return beams[0]

    def _masked_cross_entropy(self, logits, target, mask):
        """
        Computes masked cross-entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits (shape: [batch_size, seq_len, vocab_size]).
            target (torch.Tensor): Target labels (shape: [batch_size, seq_len]).
            mask (torch.Tensor): Mask for valid positions (shape: [batch_size, seq_len]).

        Returns:
            torch.Tensor: Computed masked loss.
        """
        target[~mask] = 0
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target.reshape(-1, 1)
        losses_flat = torch.gather(logits_flat, index=target_flat, dim=1)
        losses = losses_flat.reshape(*target.size())
        losses = losses * mask.float()
        loss = losses.sum() / logits.size(0)
        return loss

    def _triplet_cl(self, anchor, pos, neg, margin=0.2):
        """
        Computes the triplet contrastive loss between an anchor, positive, and negative pair.

        Args:
            anchor (torch.Tensor): The anchor tensor, typically representing
            the feature embedding of the anchor sample.
            pos (torch.Tensor): The positive tensor, typically representing
            the feature embedding of the positive sample.
            neg (torch.Tensor): The negative tensor, typically representing
            the feature embedding of the negative sample.
            margin (float, optional): The margin that is enforced between
            the anchor-positive and anchor-negative cosine similarities. Default is 0.2.

        Returns:
            torch.Tensor: The computed contrastive loss as a scalar tensor.

        Notes:
            The loss encourages the cosine similarity between the anchor and positive samples
            to be higher than the cosine similarity
            between the anchor and negative samples by at least the specified margin.
        """
        anchor_pos_cosine_similarity = nn.functional.cosine_similarity(anchor, pos, dim=-1)
        anchor_neg_cosine_similarity = nn.functional.cosine_similarity(anchor, neg, dim=-1)
        loss_cl = torch.sum(torch.max(torch.zeros_like(anchor_pos_cosine_similarity),
                                      margin - anchor_pos_cosine_similarity + anchor_neg_cosine_similarity))

        return loss_cl
