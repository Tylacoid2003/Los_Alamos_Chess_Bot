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

        # we create a LosAlamosChess() to manage the actual game and its rules
        self.board_state = LosAlamosChess()

        # Get the action dictionary from your encoder (Move String -> Int)
        self.action_dict = build_action_space()

        # Create a REVERSE dictionary (Int -> Move String)
        self.int_to_string = {idx: move for move, idx in self.action_dict.items()}

        # define action space with spaces.Discrete()
        self.action_space = spaces.Discrete(len(self.action_dict))

        # Define observation space using spaces.Box()
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (11, 6, 6),
            dtype = np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # reset game
        self.board_state.reset()

        # get 8x8 FEN of the new game
        standard_fen = self.board_state.fen()

        # get its 6x6 equivalent
        fairy_fen = to_fairy_fen(standard_fen)

        # get the tensor of fen
        observation = encode_fen(fairy_fen)
        info = {}

        return observation, info

    def step(self, action):

        # get fairy string from move
        fairy_string_move = self.int_to_string[action]

        # get standard string move
        standard_string_move = to_standard_move(fairy_string_move)

        # push move in the game
        self.board_state.push(standard_string_move)

        # get new standard FEN
        new_standard_fen = self.board_state.fen()

        # get new fairy FEN
        new_fairy_fen = to_fairy_fen(new_standard_fen)

        # get observation with encode_fen (11x6x6)
        observation = encode_fen(new_fairy_fen)

        # check if the game has ended
        terminated = self.board_state.is_game_over()
        truncated = False

        # get raw winner and convert it to a value
        raw_winner = self.board_state.winner()

        step_penalty = -0.01 # to not encourage long games 

        if raw_winner is True:
            reward = 1.0   # White wins
        elif raw_winner is False:
            reward = -1.0  # Black wins
        else:
            reward = 0.0 if terminated else step_penalty

        info = {}

        return observation, reward, terminated, truncated, info

    def get_action_mask(self):

        # create numpy array of length equal to len(self.action_dict) and make it all equal to 0
        int_legal_moves = np.zeros(len(self.action_dict), dtype=np.int8)

        # get list of current legal moves (in string format)
        current_legal_moves = self.board_state.legal_moves

        # Loop across the string_fairy_legal_moves and search for its corresponding index in self.action_dict
        for move in current_legal_moves:
            # convert standard move (in Chess.Move) to string 
            standard_string_move = move.uci()
            # convert standard string to fairy
            fairy_string_move = to_fairy_move(standard_string_move)
            # get position of the dictionary
            idx = self.action_dict[fairy_string_move]
            # update the position of the int_legal_moves
            int_legal_moves[idx] = 1

        return int_legal_moves
