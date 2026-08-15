import math
import copy
import torch
import torch.nn.functional as F
import numpy as np

from src.env.los_alamos_board import LosAlamosChess
from src.env.board_translator import to_fairy_fen
from src.env.observation_encoder import encode_fen, build_action_space
from src.utils.move_conversion import to_fairy_move

class Node:
    def __init__(self, prior: float):
        # initialize visit count, value sum, probability and empty dictionary to store node children
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # Maps move_str (e.g., 'e2e3') to Node

    @property
    def q_value(self) -> float:
        """Returns the average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device, num_simulations=50, c_puct=1.5):
        # initialize the model, device, total simulations, c_puct value, action space of LosAlamos variation and num of actions
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.action_space = build_action_space()
        self.num_actions = len(self.action_space)

    def search(self, board: LosAlamosChess) -> str:
        """Runs the MCTS loop and returns the best standard UCI move string."""
        # create root node which corresponds to the initialized board
        root = Node(prior=1.0)
        
        # Expand the root node immediately (finding the corresponding node childs of the root)
        self._evaluate_and_expand(root, board)

        for _ in range(self.num_simulations):
            # Deepcopy ensures our lookahead tree doesn't mutate the actual game board
            sim_board = copy.deepcopy(board)
            # we always start each simulation from the root
            node = root
            search_path = [node]

            # while the node that we are evaluating does not have empty child dictonary we keep going
            while node.children:
                # select move and child
                move_str, node = self._select_child(node)
                # push selected move in the virtual board
                sim_board.push(move_str)
                # append selected child into the search path
                search_path.append(node)

            # we evaluate the value (given by the value Head) of current position once the last move is made
            value = self._evaluate_and_expand(node, sim_board)

            # we update all the tree metrics in reverse order
            self._backpropagate(search_path, value)

        # we now select the move that has the highest visited count among all the node childs of the root 
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def _select_child(self, node: Node):
        """Selects the child with the highest Q + U score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        # we evaluate u and q values of each child of the current node
        for action, child in node.children.items():
            # U is the PUCT bonus pushing for exploration
            u = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            # Q is the average value found under this node
            q = child.q_value
            score = q + u

            # if the score is better that the current best, update the best score and return new best action and child
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _evaluate_and_expand(self, node: Node, board: LosAlamosChess) -> float:
        """Uses the Dual-Head CNN to evaluate the board and create child nodes."""
        # Terminal state check
        if board.is_game_over():
            winner = board.winner()
            if winner is None: return 0.0 # Draw
            # 1.0 if the player whose turn it is won, else -1.0
            return 1.0 if winner == board.board.turn else -1.0

        # Encode state for the CNN
        fairy_fen = to_fairy_fen(board.fen())
        state_array = encode_fen(fairy_fen)
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get Predictions
        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_tensor)

        # make prediction logits 1D
        policy_logits = policy_logits.squeeze(0)
        
        # Since the neural network has learned that + values = White winning and - value = Black winning. We need
        # to change the sign when it is blacks turn, so that positive means winning and negative means loosing independant
        # of the player. 
        absolute_value = value_tensor.item()
        relative_value = absolute_value if board.board.turn else -absolute_value

        # Masking and Softmax for legal moves only
        legal_moves = board.legal_moves
        legal_logits = {}
        
        for move in legal_moves:
            standard_uci = move.uci()
            fairy_uci = to_fairy_move(standard_uci)
            if fairy_uci in self.action_space:
                idx = self.action_space[fairy_uci]
                legal_logits[standard_uci] = policy_logits[idx].item()
        
        # Numerically stable softmax to convert logits to probabilities
        if legal_logits:
            max_logit = max(legal_logits.values())
            exp_priors = {k: math.exp(v - max_logit) for k, v in legal_logits.items()}
            sum_exp = sum(exp_priors.values())
            
            # Expand the node
            for action, exp_val in exp_priors.items():
                node.children[action] = Node(prior=exp_val / sum_exp)
        
        return relative_value

    def get_action_probabilities(self, board: LosAlamosChess, temperature: float = 1.0) -> tuple[int, np.ndarray]:
        """
        Runs MCTS and returns (selected_action_idx, target_policy_distribution_1356).
        Used during RL self-play data generation.
        """
        # same as search function
        root = Node(prior=1.0)
        self._evaluate_and_expand(root, board)

        for _ in range(self.num_simulations):
            sim_board = copy.deepcopy(board)
            node = root
            search_path = [node]

            while node.children:
                move_str, node = self._select_child(node)
                sim_board.push(move_str)
                search_path.append(node)

            value = self._evaluate_and_expand(node, sim_board)
            self._backpropagate(search_path, value)

        # we create an array that has the same shape as the output of the NN, we create a list with all the string moves of
        # the root node and create an array that contains the number of visits of each string move in root node
        policy_target = np.zeros(self.num_actions, dtype=np.float32)
        moves = list(root.children.keys())
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)

        if len(visits) == 0:
            return 0, policy_target

        # when we want to select the highest probability
        if temperature == 0.0:
            best_idx = np.argmax(visits)
            selected_move = moves[best_idx]
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
        else:
            # Temperature scaling on visit counts, we allow to select different moves based on the computed probability
            visits_exp = visits ** (1.0 / temperature)
            probs = visits_exp / np.sum(visits_exp)
            selected_move = np.random.choice(moves, p=probs)

        # Fill the 1356-dimensional target vector
        for move_str, prob in zip(moves, probs):
            fairy_move = to_fairy_move(move_str)
            if fairy_move in self.action_space:
                action_idx = self.action_space[fairy_move]
                policy_target[action_idx] = prob

        # save the number that corresponds to the selected string move 
        selected_fairy = to_fairy_move(selected_move)
        selected_action_idx = self.action_space[selected_fairy]

        return selected_action_idx, policy_target

    def _backpropagate(self, search_path: list, value: float):
        """Updates the visit counts and Q-values up the tree."""
        for node in reversed(search_path):
            # we flip the sign again so that from the parents prespective when it sees that one of its childs has a high
            # negative value, it means that it is good for him and back for the oponent
            value = -value
            
            node.visit_count += 1
            node.value_sum += value