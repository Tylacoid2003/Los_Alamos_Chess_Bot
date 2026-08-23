import math
import torch
import numpy as np

from src.env.los_alamos_board import LosAlamosChess
from src.env.board_translator import to_fairy_fen
from src.env.observation_encoder import encode_fen, build_action_space
from src.utils.move_conversion import to_fairy_move

class Node:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children']  # Memory & attribute lookup optimization

    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    @property
    def q_value(self) -> float:
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
        self.num_actions = len(self.action_space)

    def search(self, board: LosAlamosChess) -> str:
        root = Node(prior=1.0)
        self._evaluate_and_expand(root, board)

        for _ in range(self.num_simulations):
            # Fast native board copy
            sim_board = board.copy()
            node = root
            search_path = [node]

            while node.children:
                move_str, node = self._select_child(node)
                # Unvalidated push inside MCTS (move was already checked upon node creation)
                sim_board.push(move_str, validate=False)
                search_path.append(node)

            value = self._evaluate_and_expand(node, sim_board)
            self._backpropagate(search_path, value)

        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def _select_child(self, node: Node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        sqrt_node_visits = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            u = self.c_puct * child.prior * (sqrt_node_visits / (1 + child.visit_count))
            score = child.q_value + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _evaluate_and_expand(self, node: Node, board: LosAlamosChess) -> float:
        if board.is_game_over():
            winner = board.winner()
            if winner is None: 
                return 0.0
            return 1.0 if winner == board.board.turn else -1.0

        fairy_fen = to_fairy_fen(board.fen())
        state_array = encode_fen(fairy_fen)
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                policy_logits, value_tensor = self.model(state_tensor)

        policy_logits = policy_logits.squeeze(0)
        
        absolute_value = value_tensor.item()
        relative_value = absolute_value if board.board.turn else -absolute_value

        legal_moves = board.legal_moves
        legal_logits = {}
        
        for move in legal_moves:
            standard_uci = move.uci()
            fairy_uci = to_fairy_move(standard_uci)
            if fairy_uci in self.action_space:
                idx = self.action_space[fairy_uci]
                legal_logits[standard_uci] = policy_logits[idx].item()
        
        if legal_logits:
            max_logit = max(legal_logits.values())
            exp_priors = {k: math.exp(v - max_logit) for k, v in legal_logits.items()}
            sum_exp = sum(exp_priors.values())
            
            for action, exp_val in exp_priors.items():
                node.children[action] = Node(prior=exp_val / sum_exp)
        
        return relative_value

    def get_action_probabilities(
        self,
        board: LosAlamosChess,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.35
    ) -> tuple[int, np.ndarray]:
        root = Node(prior=1.0)
        self._evaluate_and_expand(root, board)

        if add_dirichlet_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            for i, action in enumerate(actions):
                root.children[action].prior = (
                    (1.0 - dirichlet_epsilon) * root.children[action].prior
                    + dirichlet_epsilon * noise[i]
                )

        for _ in range(self.num_simulations):
            sim_board = board.copy()
            node = root
            search_path = [node]

            while node.children:
                move_str, node = self._select_child(node)
                sim_board.push(move_str, validate=False)
                search_path.append(node)

            value = self._evaluate_and_expand(node, sim_board)
            self._backpropagate(search_path, value)

        policy_target = np.zeros(self.num_actions, dtype=np.float32)
        moves = list(root.children.keys())
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)

        if len(visits) == 0 or visits.sum() == 0:
            return 0, policy_target

        target_probs = visits / visits.sum()

        if temperature == 0.0:
            selected_move = moves[np.argmax(visits)]
        else:
            sampling_probs = visits ** (1.0 / temperature)
            sampling_probs /= sampling_probs.sum()
            selected_move = np.random.choice(moves, p=sampling_probs)

        for move_str, prob in zip(moves, target_probs):
            fairy_move = to_fairy_move(move_str)
            if fairy_move in self.action_space:
                action_idx = self.action_space[fairy_move]
                policy_target[action_idx] = prob

        selected_fairy = to_fairy_move(selected_move)
        selected_action_idx = self.action_space[selected_fairy]

        return selected_action_idx, policy_target

    def _backpropagate(self, search_path: list, value: float):
        for node in reversed(search_path):
            value = -value
            node.visit_count += 1
            node.value_sum += value