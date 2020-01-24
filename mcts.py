#!/usr/bin/env python3

import argparse
import logging
import math
import os.path
import sys
from typing import Optional, Dict, List

import torch.nn.functional
import fairseq.data
import fairseq.utils
import fairseq.checkpoint_utils


class Node:
    def __init__(self, parent=None, *, token: int, simulation: int = 0,
            depth: Optional[int] = None, probability: float = -math.inf,
            dictionary: Optional[fairseq.data.Dictionary] = None):
        self.parent: Node = parent
        self.simulation = simulation
        self.mcts_score = 0.
        self.probability = probability
        self.dictionary = \
            self.parent.dictionary if dictionary is None else dictionary
        self.children: Dict[int, Node] = {}
        self.token = token
        if depth is None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = depth

    def __repr__(self):
        c = ["%d:%s" % (x, self.dictionary[x]) for x in self.children.keys()]
        r = "Node(%s, s=%d, w=%d, d=%d, c=[%s])" % \
            (self.dictionary[self.token], self.simulation, self.mcts_score,
            self.depth, ' '.join(c))
        return r


class MonteCarloTreeSearch:
    def __init__(self, decoder: torch.nn.Module, *, eos_penalty: float = 0.0,
            dictionary: fairseq.data.Dictionary, simulation: int = 100,
            branching_factor: int = 10, exploration_factor: float = 1.0,
            cpu: bool = False):
        self.__context: Optional[torch.Tensor] = None
        self.__decoder = decoder
        self.__simulation = simulation
        self.__branching_factor = branching_factor
        self.__dictionary = dictionary
        self.__exploration_factor = exploration_factor * math.sqrt(2)
        self.__eos_penalty = eos_penalty
        self.__cpu = cpu

    def set_context(self, context: Optional[torch.Tensor]):
        self.__context = context

    def search(self):
        root = Node(token=self.__dictionary.eos(), depth=0,
            dictionary=self.__dictionary)

        cursor = self.__search_subtree(root)
        while cursor.token != self.__dictionary.eos():
            cursor = self.__search_subtree(cursor)

        output_tokens = []
        while cursor is not None:
            output_tokens.insert(0, cursor.token)
            cursor = cursor.parent
        return output_tokens

    def __search_subtree(self, root: Node):
        cursor = root
        limit = self.__simulation
        while limit > 0:
            if len(cursor.children) == 0:
                self.__expand_node(cursor)
                limit -= 1
                cursor = root
            _, cursor = self.__select_child(cursor)

        logging.debug('========================')
        logging.debug('Completed simulation, %s', root)

        key: Optional[int] = None
        node: Optional[Node] = None
        for k, v in root.children.items():
            logging.debug('Child: %s', v)
            if node is None \
                    or (v.mcts_score / v.simulation > node.mcts_score / node.simulation) \
                    or (v.mcts_score / v.simulation == node.mcts_score / node.simulation
                        and v.probability > node.probability):
                key = k
                node = v

        root.children = {
            key: Node(root, token=node.token)
        }

        logging.debug('Final choice: %s', self.__dictionary[node.token])
        return root.children[key]

    def __expand_node(self, node: Node):
        path_tokens = []
        cursor = node
        while cursor is not None:
            path_tokens.insert(0, cursor.token)
            cursor = cursor.parent
        consistent, values, indices = self.__predict(path_tokens)
        if values is not None:
            for log_prob, index in zip(values.tolist(), indices.tolist()):
                node.children[index] = \
                    Node(node, token=index, probability=log_prob)

        score = 1.0 if consistent else 0.0
        if consistent and len(path_tokens) > 2 \
                and path_tokens[-1] == self.__dictionary.eos():
            score -= self.__eos_penalty
        cursor = node
        while cursor is not None:
            cursor.simulation += 1
            cursor.mcts_score += score
            cursor = cursor.parent

    def __select_child(self, node: Node):
        selected: Optional[Node] = None
        key: Optional[int] = None
        for k, v in node.children.items():
            if self.__better(v, selected):
                selected = v
                key = k
        return key, selected

    def __better(self, first: Node, second: Optional[Node]):
        if second is None:
            return True
        if first.simulation == 0:
            if second.simulation != 0:
                return True
            return first.probability > second.probability

        # first.simulation == 0
        if second.simulation == 0:
            return False

        return self.__get_score(first) > self.__get_score(second)

    def __get_score(self, node: Node):
        avg = node.mcts_score / node.simulation
        x = math.sqrt(math.log(node.parent.simulation) / node.simulation)
        result = avg + self.__exploration_factor * x
        return result

    def __predict(self, path_tokens: List[int]):
        eos = self.__dictionary.eos()
        prev_tokens = torch.LongTensor(path_tokens + [eos]).view(1, -1)
        if not self.__cpu:
            prev_tokens = prev_tokens.cuda()
        if len(path_tokens) == 1 or path_tokens[-1] != self.__dictionary.eos():
            logits, _ = self.__decoder(prev_tokens, encoder_out=self.__context)
            logits = torch.nn.functional.log_softmax(logits[0][-1], dim=0)
            values, indices = logits.topk(self.__branching_factor)
        else:
            values, indices = None, None

        if len(path_tokens) <= 2:
            consistent = True
        else:
            consistent = self.__check_consistency(path_tokens)

        return consistent, values, indices

    def __check_consistency(self, path_tokens):
        prev_tokens = torch.LongTensor(path_tokens).view(1, -1)
        if not self.__cpu:
            prev_tokens = prev_tokens.cuda()
        logits, _ = self.__decoder(prev_tokens, encoder_out=self.__context)
        tokens = logits[0].argmax(dim=-1)
        tokens = tokens[:-1]
        assert len(tokens) + 2 == len(path_tokens)

        return tokens.tolist() == path_tokens[1:-1]


def load_checkpoint(path, data, user_dir):
    if not user_dir:
        user_dir = os.path.abspath(os.path.dirname(__file__))
    args = argparse.Namespace(user_dir=user_dir)
    fairseq.utils.import_user_module(args)
    (model,), _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [path], arg_overrides={'data': data})
    model.eval()
    return model, task


@torch.no_grad()
def mcts_decode(src_tokens, model, engine: MonteCarloTreeSearch):
    src_lengths = src_tokens.new(1, 1)
    src_lengths[0][0] = src_tokens.size(0)
    encoder_out = model.encoder(src_tokens.view(1, -1).long(), src_lengths)

    engine.set_context(encoder_out)
    output = engine.search()

    return output


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--branching-factor', type=int, default=10)
    p.add_argument('--simulation', type=int, default=100)
    p.add_argument('--user-dir', default=None)
    p.add_argument('--eos-penalty', default=0., type=float)
    p.add_argument('--exploration-factor', default=1.0, type=float)
    p.add_argument('--cpu', action='store_true', default=False)
    p.add_argument('--input', default='-')
    p.add_argument('--output', default='-')
    return p.parse_args()


def main():
    cmdline = parse_cmdline()
    logging.basicConfig(level=logging.DEBUG)

    model, task = load_checkpoint(cmdline.path, cmdline.data, cmdline.user_dir)
    if not cmdline.cpu:
        model.cuda()

    engine = MonteCarloTreeSearch(model.decoder,
        eos_penalty=cmdline.eos_penalty,
        branching_factor=cmdline.branching_factor,
        simulation=cmdline.simulation,
        exploration_factor=cmdline.exploration_factor,
        dictionary=task.target_dictionary, cpu=cmdline.cpu)

    if cmdline.input == '-':
        input_file = sys.stdin
    else:
        input_file = open(cmdline.input)
    if cmdline.output == '-':
        output_file = sys.stdout
    else:
        output_file = open(cmdline.output, 'w')

    with input_file, output_file:
        for line in input_file:
            src_tokens = task.source_dictionary.encode_line(line.strip(),
                add_if_not_exist=False)
            if not cmdline.cpu:
                src_tokens = src_tokens.cuda()
            tgt_tokens = mcts_decode(src_tokens, model, engine)
            tgt_tokens = [task.target_dictionary[x] for x in tgt_tokens]
            print(' '.join(tgt_tokens), file=output_file)


if __name__ == '__main__':
    main()
