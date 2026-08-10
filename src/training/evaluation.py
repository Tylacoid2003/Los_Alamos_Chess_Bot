import os
import csv
import torch
import torch.nn as nn

from src.env.gym_env import LosAlamosEnv
from src.models.cnn import CNN_Residual_Dual_Head_network
from src.oracle.random_openings import apply_random_opening
from src.env.board_translator import to_fairy_fen
from src.env.observation_encoder import encode_fen


def select_greedy_action(model: nn.Module, state_tensor: torch.Tensor, game: LosAlamosEnv, device: torch.device) -> int:
    """
    Selects the greedy (argmax) legal action for evaluation.
    Applies legal move masking to guarantee legal move selection.
    """
    policy_logits, _ = model(state_tensor)
    
    mask = game.get_action_mask()
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
    
    policy_masked = policy_logits.clone()
    policy_masked[:, mask_tensor == 0] = -1e9
    
    policy_prob = torch.softmax(policy_masked, dim=1)
    action = torch.argmax(policy_prob, dim=1)
    return action.item()


def evaluate_arena(
    rl_model: nn.Module, 
    baseline_model: nn.Module, 
    game: LosAlamosEnv, 
    num_games: int = 20, 
    device: torch.device = torch.device("cpu"),
    max_steps: int = 150,
    random_plies: int = 2
) -> tuple[int, int, int]:
    """
    Evaluates the RL model against a baseline model over a series of games.
    Alternates starting colors and applies random opening plies for position diversity.
    
    Returns:
        (rl_wins, baseline_wins, draws)
    """
    rl_model.eval()
    baseline_model.eval()
    
    rl_wins = 0
    baseline_wins = 0
    draws = 0

    with torch.no_grad():
        for game_idx in range(num_games):
            _, _ = game.reset()
            
            if random_plies > 0:
                apply_random_opening(game.board_state, num_plies=random_plies)
            
            # Retrieve observation tensor after applying the random opening
            current_fairy_fen = to_fairy_fen(game.board_state.fen())
            state = encode_fen(current_fairy_fen)
            
            terminated = game.board_state.is_game_over()
            truncated = False
            steps = 0

            # Alternate starting colors: Even games RL is White, Odd games RL is Black
            rl_plays_white = (game_idx % 2 == 0)

            while not (terminated or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                is_white_turn = game.board_state.board.turn

                if (is_white_turn and rl_plays_white) or (not is_white_turn and not rl_plays_white):
                    action = select_greedy_action(rl_model, state_tensor, game, device)
                else:
                    action = select_greedy_action(baseline_model, state_tensor, game, device)

                next_state, _, terminated, _, _ = game.step(action)
                state = next_state
                steps += 1

                if steps >= max_steps:
                    truncated = True

            winner = game.board_state.winner()

            if winner is True:
                if rl_plays_white:
                    rl_wins += 1
                else:
                    baseline_wins += 1
            elif winner is False:
                if not rl_plays_white:
                    rl_wins += 1
                else:
                    baseline_wins += 1
            else:
                draws += 1

    return rl_wins, baseline_wins, draws


def run_standalone_evaluation(
    rl_checkpoint_path: str, 
    baseline_checkpoint_path: str, 
    num_games: int = 100, 
    device: torch.device = torch.device("cpu")
):
    print(f"Loading RL Model Checkpoint: {rl_checkpoint_path}")
    rl_model = CNN_Residual_Dual_Head_network().to(device)
    rl_ckpt = torch.load(rl_checkpoint_path, map_location=device)
    rl_model.load_state_dict(rl_ckpt["model"])

    print(f"Loading Baseline Model Checkpoint: {baseline_checkpoint_path}")
    baseline_model = CNN_Residual_Dual_Head_network().to(device)
    base_ckpt = torch.load(baseline_checkpoint_path, map_location=device)
    baseline_model.load_state_dict(base_ckpt["model"])

    game = LosAlamosEnv()
    
    print(f"\nStarting Standalone Arena Benchmark ({num_games} games)...")
    rl_wins, base_wins, draws = evaluate_arena(
        rl_model, baseline_model, game, num_games=num_games, device=device, random_plies=2
    )

    total_games = rl_wins + base_wins + draws
    rl_winrate = (rl_wins / total_games) * 100.0

    print("\n==========================================")
    print("         STANDALONE ARENA RESULTS         ")
    print("==========================================")
    print(f"Total Games Played : {total_games}")
    print(f"RL Agent Wins      : {rl_wins} ({rl_winrate:.1f}%)")
    print(f"Baseline Wins      : {base_wins} ({(base_wins/total_games)*100.0:.1f}%)")
    print(f"Draws              : {draws} ({(draws/total_games)*100.0:.1f}%)")
    print("==========================================\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_path = "./checkpoints/RL_Finetuned_Model.pth"
    baseline_path = "./checkpoints/imitation_learning_model.pth"

    if not os.path.exists(baseline_path):
        baseline_path = "./models/Imitation_Learning_Model.pth"

    if os.path.exists(rl_path) and os.path.exists(baseline_path):
        run_standalone_evaluation(rl_path, baseline_path, num_games=100, device=device)
    else:
        print("Checkpoints not found for standalone execution. Verify file paths.")