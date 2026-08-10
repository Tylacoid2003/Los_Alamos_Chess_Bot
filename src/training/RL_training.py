import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.cnn import CNN_Residual_Dual_Head_network
from src.env.gym_env import LosAlamosEnv
from src.training.evaluation import evaluate_arena

def apply_layer_freezing(model: nn.Module, num_resblocks_to_freeze: int = 6):
    """
    Freezes lower layers (conv1, bn1, and early residual blocks)
    to protect feature extraction and prevent catastrophic forgetting.
    """
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False

    for i in range(min(num_resblocks_to_freeze, len(model.residual_blocks))):
        for param in model.residual_blocks[i].parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Layer Freezing Applied: {trainable_params}/{total_params} parameters trainable.")


def choose_action(model: nn.Module, state: torch.Tensor, game: LosAlamosEnv, device: torch.device, training=True):
    policy_logits, value = model(state)

    mask = game.get_action_mask()
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)

    policy_masked = policy_logits.clone()
    policy_masked[:, mask_tensor == 0] = -1e9

    policy_prob = torch.softmax(policy_masked, dim=1)

    if training:
        distribution = torch.distributions.Categorical(policy_prob)
        action = distribution.sample()
        entropy = distribution.entropy()
        log_prob = distribution.log_prob(action)
    else:
        with torch.no_grad():
            action = torch.argmax(policy_prob, dim=1)
            entropy = None
            log_prob = None

    return action, value, log_prob, entropy


def compute_episode_returns(trajectory: list, gamma: float = 0.99) -> list:
    G = 0.0
    for transition in reversed(trajectory):
        G = transition["reward"] + gamma * G
        transition["return"] = G
    return trajectory


def update_network_from_batch(
    optimizer: optim.Optimizer, 
    criterion_value: nn.Module, 
    batch_trajectories: list, 
    model: nn.Module, 
    gamma: float = 0.99
) -> tuple[float, float, float]:
    """
    Processes a batch of trajectories collected across multiple self-play episodes.
    Smooths out variance and stabilizes gradient updates.
    """
    all_log_probs = []
    all_values = []
    all_returns = []
    all_entropies = []
    all_turns = []

    for trajectory in batch_trajectories:
        trajectory = compute_episode_returns(trajectory, gamma=gamma)
        for transition in trajectory:
            all_log_probs.append(transition["log_prob"])
            all_values.append(transition["value"])
            all_returns.append(transition["return"])
            all_entropies.append(transition["entropy"])
            all_turns.append(transition["turn_multiplier"])

    log_probs = torch.stack(all_log_probs)
    values = torch.cat(all_values, dim=0).squeeze(1)
    returns = torch.tensor(all_returns, dtype=torch.float32, device=values.device)
    entropies = torch.stack(all_entropies)
    turns = torch.tensor(all_turns, dtype=torch.float32, device=values.device)

    # Advantage with turn-perspective correction
    advantages = (returns - values).detach() * turns

    policy_loss = -(log_probs * advantages).mean()
    value_loss = criterion_value(values, returns)
    
    c_ent = 0.01
    entropy_bonus = entropies.mean()

    total_loss = policy_loss + value_loss - (c_ent * entropy_bonus)

    optimizer.zero_grad()
    total_loss.backward()
    
    # Clip gradients across trainable parameters only
    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()), 
        max_norm=1.0
    )
    optimizer.step()

    return policy_loss.item(), value_loss.item(), total_loss.item()


def train_RL(
    model: nn.Module, 
    baseline_model: nn.Module,
    num_episodes: int, 
    optimizer: optim.Optimizer, 
    criterion_value: nn.Module,
    device: torch.device,
    batch_episodes: int = 8,
    eval_freq: int = 250,
    eval_games: int = 20
):
    print("Starting Reinforcement Learning Phase...")

    game = LosAlamosEnv()
    initial_state_space, _ = game.reset()
    state = initial_state_space

    total_episode_length = 0
    white_wins, black_wins, draws = 0, 0, 0   
    MAX_EPISODE_STEPS = 150 
    
    reports_dir = "./reports"
    checkpoints_dir = "./checkpoints"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_csv_file = os.path.join(reports_dir, "rl_training_log.csv")
    eval_csv_file = os.path.join(reports_dir, "rl_evaluation_log.csv")
    
    with open(train_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Winner", "Length", "Reward", "Policy_Loss", "Value_Loss", "Total_Loss"])

    with open(eval_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "RL_Wins", "Baseline_Wins", "Draws", "RL_Winrate"])

    batch_trajectories = []

    for episode in range(num_episodes):
        model.train()
        trajectory = []
        terminated = False
        truncated = False
        episode_length = 0

        while not (terminated or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            is_white_turn = game.board_state.board.turn
            turn_multiplier = 1.0 if is_white_turn else -1.0

            action, value, log_prob, entropy = choose_action(model, state_tensor, game, device, training=True)

            next_state, reward, terminated, _, _ = game.step(action.item())
            
            if episode_length >= MAX_EPISODE_STEPS:
                truncated = True
                reward = 0.0

            trajectory.append({
                "value": value,
                "reward": reward,
                "log_prob": log_prob,
                "entropy": entropy,
                "turn_multiplier": turn_multiplier 
            })
            
            state = next_state
            episode_length += 1

        winner = game.board_state.winner()

        if winner is True:
            white_wins += 1
            result = "White"
        elif winner is False:
            black_wins += 1
            result = "Black"
        else:
            draws += 1
            result = "Draw"

        batch_trajectories.append(trajectory)
        total_episode_length += episode_length

        # Perform optimization step once batch_episodes episodes are accumulated
        if len(batch_trajectories) >= batch_episodes or (episode + 1) == num_episodes:
            policy_loss, value_loss, total_loss = update_network_from_batch(
                optimizer=optimizer,
                criterion_value=criterion_value,
                batch_trajectories=batch_trajectories,
                model=model,
                gamma=0.99
            )
            batch_trajectories.clear()
        else:
            policy_loss, value_loss, total_loss = 0.0, 0.0, 0.0

        # Log training sample
        with open(train_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1, result, episode_length, 
                trajectory[-1]['reward'], policy_loss, value_loss, total_loss
            ])

        if (episode + 1) % 100 == 0:
            avg_len = total_episode_length / 100
            print(f"Episode {episode+1:04d}/{num_episodes} | W: {white_wins:02d} B: {black_wins:02d} D: {draws:02d} | Avg Len: {avg_len:.1f} | Last Loss: {total_loss:.4f}")
            total_episode_length = 0
            white_wins, black_wins, draws = 0, 0, 0

        # Run periodic Arena Evaluation against pre-trained model
        if (episode + 1) % eval_freq == 0:
            print(f"\n---> Running Arena Evaluation vs Pre-Trained Baseline at Episode {episode+1}...")
            rl_w, base_w, d_draws = evaluate_arena(
                rl_model=model, 
                baseline_model=baseline_model, 
                game=game, 
                num_games=eval_games, 
                device=device,
                max_steps=MAX_EPISODE_STEPS,
                random_plies=4 
            )
            
            tot_e_games = rl_w + base_w + d_draws
            winrate = (rl_w / tot_e_games) * 100.0
            print(f"---> Arena Results ({tot_e_games} games) | RL Wins: {rl_w} | Baseline Wins: {base_w} | Draws: {d_draws} | RL Winrate: {winrate:.1f}%\n")

            with open(eval_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, rl_w, base_w, d_draws, winrate])

        if (episode + 1) % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": episode + 1
                },
                os.path.join(checkpoints_dir, "RL_checkpoint.pth")
            )

        initial_state_space, _ = game.reset()
        state = initial_state_space

    print("Training completed. Saving final fine-tuned model...")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "./models/RL_Finetuned_Model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing RL Training on device: {device}")

    pretrained_path = "./checkpoints/imitation_learning_model.pth"
    if not os.path.exists(pretrained_path):
        pretrained_path = "./models/Imitation_Learning_Model.pth"

    model = CNN_Residual_Dual_Head_network().to(device)
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Apply Layer Freezing
    apply_layer_freezing(model, num_resblocks_to_freeze=6)

    # Baseline Model for Arena Evaluation
    baseline_model = CNN_Residual_Dual_Head_network().to(device)
    baseline_model.load_state_dict(checkpoint["model"])
    baseline_model.eval()

    num_episodes = 2500
    # Pass only trainable parameters to optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-5, 
        weight_decay=1e-4
    )
    criterion_value = nn.SmoothL1Loss()

    train_RL(
        model=model, 
        baseline_model=baseline_model, 
        num_episodes=num_episodes, 
        optimizer=optimizer, 
        criterion_value=criterion_value,
        device=device,
        batch_episodes=8,
        eval_freq=250,
        eval_games=20
    )