import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.env.los_alamos_board import LosAlamosChess
from src.env.observation_encoder import build_action_space, encode_fen
from src.utils.move_conversion import to_fairy_move, to_standard_move
from src.env.board_translator import to_fairy_fen

class LosAlamosEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_state = LosAlamosChess()
        self.action_dict = build_action_space()
        self.int_to_string = {idx: move for move, idx in self.action_dict.items()}
        self.action_space = spaces.Discrete(len(self.action_dict))

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11, 6, 6),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.board_state.reset()
        
        standard_fen = self.board_state.fen()
        fairy_fen = to_fairy_fen(standard_fen)
        observation = encode_fen(fairy_fen)
        
        return observation, {}

    def step(self, action):
        fairy_string_move = self.int_to_string[action]
        standard_string_move = to_standard_move(fairy_string_move)

        self.board_state.push(standard_string_move)

        new_standard_fen = self.board_state.fen()
        new_fairy_fen = to_fairy_fen(new_standard_fen)
        observation = encode_fen(new_fairy_fen)

        terminated = self.board_state.is_game_over()
        truncated = False
        raw_winner = self.board_state.winner()

        if raw_winner is True:
            reward = 1.0   # White wins
        elif raw_winner is False:
            reward = -1.0  # Black wins
        else:
            reward = 0.0   # Ongoing or Draw

        return observation, reward, terminated, truncated, {}

    def get_action_mask(self):
        int_legal_moves = np.zeros(len(self.action_dict), dtype=np.int8)
        current_legal_moves = self.board_state.legal_moves

        for move in current_legal_moves:
            standard_string_move = move.uci()
            fairy_string_move = to_fairy_move(standard_string_move)
            if fairy_string_move in self.action_dict:
                idx = self.action_dict[fairy_string_move]
                int_legal_moves[idx] = 1

        return int_legal_moves