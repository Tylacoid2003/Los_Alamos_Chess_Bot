import math
import copy
import torch
import torch.nn.functional as F

from src.env.los_alamos_board import LosAlamosChess
from src.env.board_translator import to_fairy_fen
from src.env.observation_encoder import encode_fen, build_action_space
from src.utils.move_conversion import to_fairy_move

class Node:
    def __init__(self, prior: float):
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
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.action_space = build_action_space()

    def search(self, board: LosAlamosChess) -> str:
        """Runs the MCTS loop and returns the best standard UCI move string."""
        root = Node(prior=1.0)
        
        # 1. Expand the root node immediately
        self._evaluate_and_expand(root, board)

        for _ in range(self.num_simulations):
            # Deepcopy ensures our lookahead tree doesn't mutate the actual game board
            sim_board = copy.deepcopy(board)
            node = root
            search_path = [node]

            # 2. Selection: Traverse down to an unexpanded leaf node
            while node.children:
                move_str, node = self._select_child(node)
                sim_board.push(move_str)
                search_path.append(node)

            # 3. Expansion & Evaluation: Score the leaf with the Neural Net
            value = self._evaluate_and_expand(node, sim_board)

            # 4. Backpropagation: Update tree metrics
            self._backpropagate(search_path, value)

        # 5. Action Selection: Pick the move heavily favored by the simulation
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def _select_child(self, node: Node):
        """Selects the child with the highest Q + U score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # U is the PUCT bonus pushing for exploration
            u = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            # Q is the average value found under this node
            q = child.q_value
            score = q + u

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
        
        policy_logits = policy_logits.squeeze(0)
        
        # Convert absolute value (White=+, Black=-) to relative value for the current turn
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

    def _backpropagate(self, search_path: list, value: float):
        """Updates the visit counts and Q-values up the tree."""
        for node in reversed(search_path):
            # FLIP FIRST: The value returned is from the perspective of the leaf node's player.
            # We must flip it so the child node stores the value from the PARENT's perspective.
            value = -value
            
            node.visit_count += 1
            node.value_sum += value