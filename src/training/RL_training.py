import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from src.models.cnn import CNN_Residual_Dual_Head_network
from src.env.gym_env import LosAlamosEnv
from src.training.mcts import MCTS
from src.training.evaluation import evaluate_arena
from src.utils.move_conversion import to_fairy_move

# ---------- Layer Freezing (early layers) ----------
def apply_layer_freezing(model: nn.Module, num_resblocks_to_freeze: int = 4):
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
    return model

# ---------- Training on a Batch from Replay Buffer ----------
def train_on_batch(
    model: nn.Module,
    baseline_model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler, # Passed in from main loop
    criterion_value: nn.Module,
    replay_buffer: deque,
    device: torch.device,
    batch_size: int = 256,
    num_gradient_steps: int = 8,
    distill_weight: float = 0.0
):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    model.train()
    model.conv1.eval()
    model.bn1.eval()
    for i in range(min(4, len(model.residual_blocks))):
        model.residual_blocks[i].eval()

    total_p_loss, total_v_loss, total_t_loss = 0.0, 0.0, 0.0
    total_mcts_entropy, total_net_entropy, total_grad_norm = 0.0, 0.0, 0.0

    for _ in range(num_gradient_steps):
        batch = random.sample(replay_buffer, batch_size)

        states = torch.tensor(np.array([d[0] for d in batch]), dtype=torch.float32, device=device)
        target_pis = torch.tensor(np.array([d[1] for d in batch]), dtype=torch.float32, device=device)
        target_vs = torch.tensor(np.array([d[2] for d in batch]), dtype=torch.float32, device=device).unsqueeze(1)

        with torch.amp.autocast('cuda'):
            policy_pred, value_pred = model(states)

            # Policy loss
            log_probs = F.log_softmax(policy_pred, dim=1)
            policy_loss = -(target_pis * log_probs).sum(dim=1).mean()

            # Value loss
            value_loss = criterion_value(value_pred, target_vs)

            # Distillation loss (Safely bypassed if 0.0)
            distill_loss = 0.0
            if distill_weight > 0.0:
                with torch.no_grad():
                    baseline_policy_logits, _ = baseline_model(states)
                baseline_probs = F.softmax(baseline_policy_logits, dim=1)
                distill_loss = F.kl_div(log_probs, baseline_probs, reduction='batchmean')

            total_loss = policy_loss + value_loss + (distill_weight * distill_loss)

        # Fix: Proper AMP Gradient Clipping Order
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()

        # Diagnostics: Dual Entropy Calculation
        mcts_entropy = -(target_pis * torch.log(target_pis + 1e-9)).sum(dim=1).mean()
        net_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=1).mean()

        total_p_loss += policy_loss.item()
        total_v_loss += value_loss.item()
        total_t_loss += total_loss.item()
        total_mcts_entropy += mcts_entropy.item()
        total_net_entropy += net_entropy.item()
        total_grad_norm += grad_norm.item()

    return (total_p_loss / num_gradient_steps,
            total_v_loss / num_gradient_steps,
            total_t_loss / num_gradient_steps,
            total_mcts_entropy / num_gradient_steps,
            total_net_entropy / num_gradient_steps,
            total_grad_norm / num_gradient_steps)

# ---------- Main RL Loop ----------
def train_rl_alpha_zero(
    model: nn.Module,
    baseline_model: nn.Module,
    num_episodes: int = 400,
    total_planned_episodes=None,
    mcts_sims: int = 250,
    batch_size: int = 256,
    num_gradient_steps: int = 8,
    batch_episodes: int = 8,
    eval_freq: int = 50,
    eval_games: int = 20,
    device: torch.device = torch.device("cuda"),
    replay_buffer_capacity: int = 100000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    distill_weight: float = 0.0,
    max_steps_per_episode: int = 150
):
    print("Starting AlphaZero-style RL fine‑tuning (Diagnostic Run)")
    torch.backends.cudnn.benchmark = True

    env = LosAlamosEnv()
    mcts = MCTS(model=model, device=device, num_simulations=mcts_sims, c_puct=1.5)
    replay_buffer = deque(maxlen=replay_buffer_capacity)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize scaler ONCE here
    scaler = torch.amp.GradScaler('cuda')
    criterion_value = nn.SmoothL1Loss()

    total_planned_episodes = total_planned_episodes or num_episodes
    total_updates = total_planned_episodes // batch_episodes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)

    os.makedirs("./reports", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_csv = "./reports/rl_400_train.csv"
    eval_csv = "./reports/rl_400_eval.csv"

    with open(train_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Winner", "Length", "Was_truncated", "Policy_Loss", "Value_Loss", "Total_Loss", "MCTS_Ent", "Net_Ent", "Grad_Norm"])

    with open(eval_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "RL_Wins", "Baseline_Wins", "Draws", "RL_Winrate", "Truncated"])

    last_p_loss = last_v_loss = last_t_loss = last_m_ent = last_n_ent = last_g_norm = 0.0

    white_wins = black_wins = draws = 0
    total_len = 0

    print("Entering main loop...")
    for episode in range(1, num_episodes + 1):
        model.eval()

        env.reset()
        state, _ = env.reset()
        trajectory = [] 
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            temp = 1.0 if steps < 10 else 0.0

            action_idx, target_pi = mcts.get_action_probabilities(
                env.board_state,
                temperature=temp,
                add_dirichlet_noise=True,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.35
            )

            trajectory.append((state.copy(), target_pi))
            next_state, reward, done, truncated, _ = env.step(action_idx)
            state = next_state
            steps += 1

            if steps >= max_steps_per_episode:
                truncated = True
                break

        winner = env.board_state.winner()
        if winner is True:
            z = 1.0
            white_wins += 1
            result = "White"
        elif winner is False:
            z = -1.0
            black_wins += 1
            result = "Black"
        else:
            z = 0.0
            draws += 1
            result = "Draw"

        for s, pi in trajectory:
            replay_buffer.append((s, pi, z))

        total_len += steps

        if episode % batch_episodes == 0 and len(replay_buffer) >= batch_size:
            last_p_loss, last_v_loss, last_t_loss, last_m_ent, last_n_ent, last_g_norm = train_on_batch(
                model=model,
                baseline_model=baseline_model,
                optimizer=optimizer,
                scaler=scaler,
                criterion_value=criterion_value,
                replay_buffer=replay_buffer,
                device=device,
                batch_size=batch_size,
                num_gradient_steps=num_gradient_steps,
                distill_weight=distill_weight
            )
            scheduler.step()

        was_truncated = truncated and not done

        with open(train_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([episode, result, steps, was_truncated, last_p_loss, last_v_loss, last_t_loss, last_m_ent, last_n_ent, last_g_norm])

        if episode % 10 == 0:
            avg_len = total_len / 10 if episode % 10 == 0 else 0
            d_kl = last_p_loss - last_m_ent
            print(f"Ep {episode:04d} | W:{white_wins:02d} B:{black_wins:02d} D:{draws:02d} | Len:{avg_len:.1f} | "
                  f"P_Loss:{last_p_loss:.3f} (D_KL:{d_kl:.3f}) V_Loss:{last_v_loss:.3f} | M_Ent:{last_m_ent:.3f} N_Ent:{last_n_ent:.3f} | G_Norm:{last_g_norm:.3f}")
            white_wins = black_wins = draws = 0
            total_len = 0

        if episode % eval_freq == 0:
            print(f"\n---> Arena Evaluation @ Episode {episode}")
            rl_w, base_w, d_draws, truncated_games = evaluate_arena(
                rl_model=model,
                baseline_model=baseline_model,
                num_games=eval_games,
                device=device,
                mcts_sims=250,
                opening_plies=4
            )
            tot = rl_w + base_w + d_draws
            winrate = (rl_w / tot) * 100.0 if tot > 0 else 0.0
            trunc_pct = (truncated_games/eval_games)*100
            print(f"Results: RL {rl_w} | Baseline {base_w} | Draws {d_draws} | Winrate {winrate:.1f}% | Truncated {trunc_pct:.1f}%\n")
            
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode
            }, f"./checkpoints/rl_400_ep{episode}.pth")

            with open(eval_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([episode, rl_w, base_w, d_draws, winrate, trunc_pct])

    torch.save({"model": model.state_dict()}, "./models/RL_400_Final.pth")
    print("Check completed. Final model saved.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    pretrained_path = "./models/Imitation_Learning_Model2.pth"
    if not os.path.exists(pretrained_path):
        pretrained_path = "./checkpoints/imitation_learning_model.pth"

    checkpoint = torch.load(pretrained_path, map_location=device)

    model = CNN_Residual_Dual_Head_network().to(device)
    model.load_state_dict(checkpoint["model"])
    apply_layer_freezing(model, num_resblocks_to_freeze=4)

    baseline_model = CNN_Residual_Dual_Head_network().to(device)
    baseline_model.load_state_dict(checkpoint["model"])
    baseline_model.eval()

    train_rl_alpha_zero(
        model=model,
        baseline_model=baseline_model,
        num_episodes=200,              # Short test run
        total_planned_episodes=400,
        mcts_sims=250,
        batch_size=256,               
        num_gradient_steps=8,         # High update density
        batch_episodes=8,
        eval_freq=25,
        eval_games=40,
        device=device,
        replay_buffer_capacity=30000,
        learning_rate=1e-4,
        weight_decay=1e-4,
        distill_weight=0.0,           # Pure RL isolation
        max_steps_per_episode=150
    )