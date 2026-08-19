import torch
import torch.nn.functional as F
import numpy as np
import random
import chess

# Imports matching the provided Los_Alamos_Chess_Bot architecture
from src.env.los_alamos_board import LosAlamosChess
from src.env.board_translator import to_fairy_fen
from src.utils.move_conversion import to_fairy_move, to_standard_move
from src.env.observation_encoder import encode_fen, build_action_space
from src.models.cnn import CNN_Residual_Dual_Head_network
from src.training.mcts import Node, MCTS 

class RandomAgent:
    """Baseline 1: Sanity Check - Plays purely random legal moves."""
    def select_move(self, board: LosAlamosChess):
        # We extract the UCI string to maintain consistency with board.push()
        return random.choice(board.legal_moves).uci()

class GreedyAgent:
    """Baseline 2: Intermediate - Prioritizes capturing material."""
    def select_move(self, board: LosAlamosChess):
        # Access the underlying python-chess board to check for captures
        captures = [m for m in board.legal_moves if board.board.is_capture(m)]
        if captures:
            return random.choice(captures).uci()
        return random.choice(board.legal_moves).uci()

class PreTrainedAgent:
    """Your Dual-Head CNN AlphaZero Model (Raw Intuition)."""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {self.device}")
                
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        # Load Model Architecture
        self.model = CNN_Residual_Dual_Head_network().to(self.device)
        
        # Load the saved file from disk
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check if it's a full training checkpoint or just a raw state_dict
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

        # Build Action Mappings
        self.action_space = build_action_space()
        self.index_to_action = {idx: move for move, idx in self.action_space.items()}

    def select_move(self, board: LosAlamosChess, temperature=0.0):
        # 1. Parse standard 8x8 FEN -> 6x6 Fairy FEN -> (11, 6, 6) numpy array
        fairy_fen = to_fairy_fen(board.fen())
        state_array = encode_fen(fairy_fen)
        
        # Convert numpy array to tensor and add batch dimension
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
        
        policy_logits = policy_logits.squeeze(0) # Shape: (1356,)

        # 2. Action Masking: Initialize mask with negative infinity
        masked_logits = torch.full_like(policy_logits, float('-inf'))

        # 3. Apply the mask for legal actions
        legal_moves = board.legal_moves
        for move in legal_moves:
            standard_uci = move.uci()
            fairy_uci = to_fairy_move(standard_uci)
            
            if fairy_uci in self.action_space:
                idx = self.action_space[fairy_uci]
                masked_logits[idx] = policy_logits[idx]

        # 4. Select Action
        if temperature == 0.0:
            # Deterministic: pick the max legal logit
            best_idx = torch.argmax(masked_logits).item()
        else:
            # Stochastic: sample from the valid probability distribution
            probs = F.softmax(masked_logits / temperature, dim=0)
            best_idx = torch.multinomial(probs, 1).item()

        # 5. Translate back to standard UCI for LosAlamosChess.push()
        best_fairy_move = self.index_to_action[best_idx]
        return to_standard_move(best_fairy_move)

class MCTSAgent:
    """Your MCTS-wrapped Dual-Head CNN AlphaZero Model (Lookahead Calculation)."""
    def __init__(self, model_path, device='cuda', num_simulations=50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load Model Architecture
        self.model = CNN_Residual_Dual_Head_network().to(self.device)
        
        # Load the saved file from disk
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check if it's a full training checkpoint or just a raw state_dict
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

        # Initialize the Monte Carlo Tree Search class
        self.mcts = MCTS(model=self.model, device=self.device, num_simulations=num_simulations, c_puct=1.5)

    def select_move(self, board: LosAlamosChess):
        # The MCTS search completely handles the logic and directly returns the best standard UCI move string
        return self.mcts.search(board)

class Arena:
    """Evaluation loop ensuring alternating colors."""
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b

    def play_game(self, white_agent, black_agent):
        board = LosAlamosChess()
        
        while not board.is_game_over():
            # python-chess turn: True = White, False = Black
            if board.board.turn:
                move_str = white_agent.select_move(board)
            else:
                move_str = black_agent.select_move(board)
                
            board.push(move_str)
            
        return board.winner() 

    def evaluate(self, num_games=100):
        print(f"Starting Arena Evaluation: {num_games} Games\n")
        results = {"model_wins": 0, "baseline_wins": 0, "draws": 0}
        
        for i in range(num_games):
            # Alternate colors to neutralize first-mover advantage
            if i % 2 == 0:
                # Model plays as White (True)
                winner = self.play_game(white_agent=self.agent_a, black_agent=self.agent_b)
                if winner is True: results["model_wins"] += 1
                elif winner is False: results["baseline_wins"] += 1
                else: results["draws"] += 1
            else:
                # Model plays as Black (False)
                winner = self.play_game(white_agent=self.agent_b, black_agent=self.agent_a)
                if winner is False: results["model_wins"] += 1
                elif winner is True: results["baseline_wins"] += 1
                else: results["draws"] += 1
                
            print(f"Game {i + 1:03d}/{num_games} Complete | Model Wins: {results['model_wins']:03d} | Baseline Wins: {results['baseline_wins']:03d} | Draws: {results['draws']:03d}")
                
        return results

if __name__ == "__main__":
    # Ensure correct pathing from your project root
    model_path = "./models/Imitation_Learning_Model2.pth"
    
    print("Loading MCTS Agent...")
    # Wrap the model in the MCTS Agent instead of the raw PreTrainedAgent
    mcts_agent = PreTrainedAgent(model_path=model_path)
    
    print("Initializing Baseline...")
    random_agent = RandomAgent()
    
    # Pit the MCTS Agent against the Random Agent
    arena = Arena(agent_a=mcts_agent, agent_b=random_agent)
    
    final_stats = arena.evaluate(num_games=500)
    
    print("\n--- Final Results ---")
    print(f"Pre-trained Model vs Random Agent Results: {final_stats}")