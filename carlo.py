import chess
from copy import deepcopy
import math
import random
from statistics import mean
from time import perf_counter, perf_counter_ns

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 305,
    chess.BISHOP: 333,
    chess.ROOK: 563,
    chess.QUEEN: 950,
    chess.KING: 10_000,
}


class Node:
    def __init__(self, move, parent):  # move is from parent to node
        self.move = move
        self.parent = parent
        self.children = []
        self.score = 0
        self.visits = 0

    def expand_node(self, state):
        if not state.is_game_over():
            for move in state.legal_moves:
                new_child = Node(move, self)
                self.children.append(new_child)

    def update(self, result):
        self.visits += 1
        self.score += result

    def is_leaf(self):
        return len(self.children) == 0

    def has_parent(self):
        return self.parent is not None


def node_comparator(node):
    if node.visits == 0:
        return 0.5
    return node.score / node.visits


def tree_policy_child(node, is_maximizing_player):
    if node.is_leaf():
        return node

    if is_maximizing_player:
        best_uct = -float("inf")
    else:
        best_uct = float("inf")
    best_node = None

    for child in node.children:
        if child.visits == 0:
            return child

        child_ratio = child.score / child.visits

        if is_maximizing_player:
            uct_value = child_ratio + math.sqrt(
                2 * math.log(node.visits) / child.visits
            )
            if uct_value > best_uct:
                best_uct = uct_value
                best_node = child
        else:
            uct_value = child_ratio - math.sqrt(
                2 * math.log(node.visits) / child.visits
            )
            if uct_value < best_uct:
                best_uct = uct_value
                best_node = child

    return best_node


def captured_piece(state, move):
    if state.is_en_passant(move):
        return chess.PAWN
    else:
        return state.piece_at(move.to_square).piece_type


def material_balance(piece_map):
    diff = 0
    for piece in piece_map.values():
        if piece.color:
            diff += PIECE_VALUES[piece.piece_type]
        else:
            diff -= PIECE_VALUES[piece.piece_type]

    return 1 / (1 + math.exp(-diff / 200))


def simulation_policy_child(state):
    while not state.is_game_over():
        p_map = state.piece_map()
        if len(p_map) <= 6:
            return material_balance(p_map)

        move_list = []
        best_capture = None
        best_captured_piece = -1
        for move in state.legal_moves:
            move_list.append(move)
            if state.is_capture(move):
                captured = captured_piece(state, move)
                if captured > best_captured_piece:
                    best_capture = move
                    best_captured_piece = captured

        if best_capture is not None:
            choice_move = best_capture
        else:
            choice_move = random.choice(move_list)
        state.push(choice_move)

    outcome = state.outcome().winner
    if outcome == chess.WHITE:
        return 1
    elif outcome == chess.BLACK:
        return 0
    else:
        return 0.5


def best_move(root, is_white):
    root.children.sort(key=node_comparator)
    if is_white:
        return root.children[-1].move
    else:
        return root.children[0].move


def mcts(root_state, time_out):
    root_node = Node(None, None)

    start = perf_counter()
    count = 0
    while (perf_counter() - start) < time_out:
        count += 1
        if count % 1000 == 0:
            elapsed_time = perf_counter() - start
            print(f"""
            Elapsed time: {elapsed_time // 60} min and {elapsed_time % 60:.2f} s
            Iters: {count} it
            Rate: {(count / elapsed_time):.2f} it/s
            """)

        node, state = root_node, deepcopy(root_state)

        while not node.is_leaf():  # select leaf
            node = tree_policy_child(node, state.turn)
            state.push(node.move)

        node.expand_node(state)  # expand
        node = tree_policy_child(node, state.turn)

        result = simulation_policy_child(state)  # simulate

        while node.has_parent():  # propagate
            node.update(result)
            node = node.parent
        root_node.update(result)

    print(f"Runs: {count}")
    root_node.children.sort(key=node_comparator)
    for top_kid in root_node.children:
        print(top_kid.move, node_comparator(top_kid))

    return best_move(root_node, root_state.turn)


# start_fen = "rnb1kb1r/ppp1pppp/5n2/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1"
start_fen = "rnbqkb1r/ppp1pppp/8/8/2n5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
origin = chess.Board(fen=start_fen)
print("Chose:", mcts(origin, 60))
