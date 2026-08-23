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
    def select_move(self, board: LosAlamosChess, **kwargs):  # <-- ADD **kwargs
        return random.choice(board.legal_moves).uci()

class GreedyAgent:
    """Baseline 2: Intermediate - Prioritizes capturing material."""
    def select_move(self, board: LosAlamosChess, **kwargs):  # <-- ADD **kwargs
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
    def __init__(self, model_path=None, model=None, device='cuda', num_simulations=50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Scenario A: An already-loaded model was passed directly (used during RL training evaluation)
        if model is not None:
            self.model = model.to(self.device)
            
        # Scenario B: A file path was provided (used during normal inference/play)
        elif model_path is not None:
            # Load Model Architecture
            self.model = CNN_Residual_Dual_Head_network().to(self.device)
            
            # Load the saved file from disk
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Check if it's a full training checkpoint or just a raw state_dict
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise ValueError("You must provide either a 'model_path' or a 'model' object.")
            
        self.model.eval()

        # Initialize the Monte Carlo Tree Search class
        self.mcts = MCTS(model=self.model, device=self.device, num_simulations=num_simulations, c_puct=1.5)

    def select_move(self, board: LosAlamosChess, temperature: float = 0.0):
        # If temperature > 0, sample probabilistically for opening diversity
        if temperature > 0.0:
            action_idx, _ = self.mcts.get_action_probabilities(
                board,
                temperature=temperature,
                add_dirichlet_noise=False  # Keep search clean, but sample top visits
            )
            # Find the move string from the selected index
            fairy_move = [move for move, idx in self.mcts.action_space.items() if idx == action_idx][0]
            return to_standard_move(fairy_move)
        
        # Deterministic play (greedy argmax)
        return self.mcts.search(board)
class Arena:
    """Evaluation loop ensuring alternating colors."""
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b

    def play_game(self, white_agent, black_agent, max_steps=150, opening_plies=4):
        board = LosAlamosChess()
        steps = 0
        
        while not board.is_game_over() and steps < max_steps:
            # Add temperature to the first few plies to create unique opening branches
            temp = 1.0 if steps < opening_plies else 0.0
            
            if board.board.turn:
                move_str = white_agent.select_move(board, temperature=temp)
            else:
                move_str = black_agent.select_move(board, temperature=temp)
                
            board.push(move_str)
            steps += 1

        was_truncated = steps >= max_steps and not board.is_game_over()
        return board.winner(), was_truncated

    def evaluate(self, num_games=100, opening_plies=4):
        print(f"Starting Arena Evaluation: {num_games} Games\n")
        results = {"model_wins": 0, "baseline_wins": 0, "draws": 0, "truncated": 0}
        
        for i in range(num_games):
            # Alternate colors to neutralize first-mover advantage
            if i % 2 == 0:
                # Model plays as White (True)
                winner, was_truncated = self.play_game(white_agent=self.agent_a, black_agent=self.agent_b, opening_plies=opening_plies)
                if winner is True: results["model_wins"] += 1
                elif winner is False: results["baseline_wins"] += 1
                else: results["draws"] += 1
                
            else:
                # Model plays as Black (False)
                winner, was_truncated = self.play_game(white_agent=self.agent_b, black_agent=self.agent_a, opening_plies=opening_plies)
                if winner is False: results["model_wins"] += 1
                elif winner is True: results["baseline_wins"] += 1
                else: results["draws"] += 1

            if was_truncated is True: results["truncated"] += 1
            # print(f"Game {i + 1:03d}/{num_games} Complete | Model Wins: {results['model_wins']:03d} | Baseline Wins: {results['baseline_wins']:03d} | Draws: {results['draws']:03d}")
                
        return results

def evaluate_arena(
    rl_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    num_games: int = 20,
    device: str = 'cuda',
    mcts_sims: int = 50,
    opening_plies: int = 4  # <-- Add this parameter
) -> tuple[int, int, int, int]:
    
    rl_agent = MCTSAgent(model=rl_model, device=device, num_simulations=mcts_sims)
    base_agent = MCTSAgent(model=baseline_model, device=device, num_simulations=mcts_sims)

    arena = Arena(agent_a=rl_agent, agent_b=base_agent)
    
    # Pass the opening plies to the evaluate method
    stats = arena.evaluate(num_games=num_games, opening_plies=opening_plies) 
    return stats["model_wins"], stats["baseline_wins"], stats["draws"], stats["truncated"]
if __name__ == "__main__":
    pre_model_path = "./models/Imitation_Learning_Model2.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading models and heuristic agents...")
    pre_trained_agent = PreTrainedAgent(model_path=pre_model_path, device=device)
    mcts_agent = MCTSAgent(model_path=pre_model_path, device=device, num_simulations=50)
    
    random_agent = RandomAgent()
    greedy_agent = GreedyAgent()
    
    num_games = 200
    
    # We force opening_plies=0 here so that the neural networks act deterministically 
    # for this specific heuristic benchmark.
    benchmark_opening_plies = 0

    print("\n" + "="*60)
    print(f"BENCHMARK 1: Pre-Trained Model vs Random Agent ({num_games} Games)")
    print("="*60)
    arena_1 = Arena(agent_a=pre_trained_agent, agent_b=random_agent)
    stats_1 = arena_1.evaluate(num_games=num_games, opening_plies=benchmark_opening_plies)

    print("\n" + "="*60)
    print(f"BENCHMARK 2: Pre-Trained Model vs Greedy Agent ({num_games} Games)")
    print("="*60)
    arena_2 = Arena(agent_a=pre_trained_agent, agent_b=greedy_agent)
    stats_2 = arena_2.evaluate(num_games=num_games, opening_plies=benchmark_opening_plies)

    print("\n" + "="*60)
    print(f"BENCHMARK 3: Pre-Trained + MCTS (50 sim) vs Random Agent ({num_games} Games)")
    print("="*60)
    arena_3 = Arena(agent_a=mcts_agent, agent_b=random_agent)
    stats_3 = arena_3.evaluate(num_games=num_games, opening_plies=benchmark_opening_plies)

    print("\n" + "="*60)
    print(f"BENCHMARK 4: Pre-Trained + MCTS (50 sim) vs Greedy Agent ({num_games} Games)")
    print("="*60)
    arena_4 = Arena(agent_a=mcts_agent, agent_b=greedy_agent)
    stats_4 = arena_4.evaluate(num_games=num_games, opening_plies=benchmark_opening_plies)

    matchups = [
        ("Pre-Trained vs Random", stats_1),
        ("Pre-Trained vs Greedy", stats_2),
        ("Pre-Trained + MCTS (50) vs Random", stats_3),
        ("Pre-Trained + MCTS (50) vs Greedy", stats_4),
    ]

    print("\n" + "#"*70)
    print("SUMMARY FOR TABLE")
    print("#"*70)
    print(f"{'Configuration':<35} | {'Win %':<8} | {'Loss %':<8} | {'Draw %':<8} | {'Raw (W-L-D)'}")
    print("-" * 75)
    for name, s in matchups:
        w_pct = (s['model_wins'] / num_games) * 100
        l_pct = (s['baseline_wins'] / num_games) * 100
        d_pct = (s['draws'] / num_games) * 100
        raw_str = f"{s['model_wins']}-{s['baseline_wins']}-{s['draws']}"
        print(f"{name:<35} | {w_pct:>6.1f}% | {l_pct:>6.1f}% | {d_pct:>6.1f}% | {raw_str}")