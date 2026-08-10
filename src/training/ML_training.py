import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from src.env.observation_encoder import encode_fen, build_action_space
from src.models.cnn import CNN_Residual_Dual_Head_network, ResidualBlock
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

class LosAlamosDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        with h5py.File(h5_path, "r") as h5file:
            self.fens = [fen.decode('utf-8') for fen in h5file["fens"][:]]
            self.moves = [move.decode('utf-8') for move in h5file["moves"][:]]
            self.values = h5file["values"][:]
            self.winners = h5file["winners"][:]

        self.action_space = build_action_space()
        self.policy_indices = [
            self.action_space[m]
            for m in self.moves
        ]

    def __len__(self):
        return len(self.fens)
    
    def __getitem__(self, idx):
        state_space_tensor = torch.from_numpy(
            encode_fen(self.fens[idx])
        ).float()

        policy_target_tensor = torch.tensor(
            self.policy_indices[idx],
            dtype=torch.long
        )

        # Relaxed Normalization
        raw_stockfish = float(self.values[idx])

        turn_indicator = self.fens[idx].split()[1]   # 'w' or 'b'

        if turn_indicator == 'b':
            raw_stockfish = -raw_stockfish

        normalized_stockfish = np.tanh(raw_stockfish / 600.0) 
        
        # Blending Stockfish Evaluation with True Game Outcome
        game_outcome = float(self.winners[idx])
        alpha = 0.8

        blended_value = alpha * normalized_stockfish + (1 - alpha) * game_outcome
                
        normalized_values_tensor = torch.tensor([blended_value], dtype=torch.float32)

        return state_space_tensor, policy_target_tensor, normalized_values_tensor


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion_policy, criterion_value, device: torch.device):
    model.eval()
    running_total, running_policy, running_value = 0.0, 0.0, 0.0
    running_top1, running_top3, running_top5 = 0, 0, 0
    
    with torch.no_grad():
        for data in dataloader:
            state, pol_targ, val_targ = data
            state, pol_targ, val_targ = state.to(device), pol_targ.to(device), val_targ.to(device)

            pol_pred, val_pred = model(state)

            pol_loss = criterion_policy(pol_pred, pol_targ)
            val_loss = criterion_value(val_pred, val_targ)
            total_loss = pol_loss + (20.0 * val_loss)

            running_total += total_loss.item()
            running_policy += pol_loss.item()
            running_value += val_loss.item()
            
            # Top-K Accuracy
            _, top_preds = pol_pred.topk(5, dim=1)
            target_expanded = pol_targ.unsqueeze(1)
            running_top1 += (top_preds[:, :1] == target_expanded).sum().item()
            running_top3 += (top_preds[:, :3] == target_expanded).sum().item()
            running_top5 += (top_preds[:, :5] == target_expanded).sum().item()
            
    num_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    
    avg_total = running_total / num_batches
    avg_pol = running_policy / num_batches
    avg_val = running_value / num_batches
    
    acc_top1 = (running_top1 / total_samples) * 100.0
    acc_top3 = (running_top3 / total_samples) * 100.0
    acc_top5 = (running_top5 / total_samples) * 100.0
    
    return avg_total, avg_pol, avg_val, acc_top1, acc_top3, acc_top5


def train_dual_head_network(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader,     
    test_dataloader: DataLoader,    
    criterion_policy: nn.Module, 
    criterion_value: nn.Module, 
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int, 
    device: torch.device,
    patience: int = 10           
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[int]]:
    
    history_train_total, history_train_policy, history_train_value = [], [], []
    history_val_total, history_val_policy, history_val_value = [], [], []
    history_epochs = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_path = "./models/Imitation_Learning_Model.pth"

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")

        # --- TRAINING PHASE ---
        model.train()
        running_train_total, running_train_policy, running_train_value = 0.0, 0.0, 0.0
        running_train_top1, running_train_top3, running_train_top5 = 0, 0, 0
        
        for i, data in enumerate(train_dataloader):
            state_space, policy_target, value_target = data
            state_space, policy_target, value_target = state_space.to(device), policy_target.to(device), value_target.to(device)

            optimizer.zero_grad()
            policy_pred, value_pred = model(state_space)

            policy_loss = criterion_policy(policy_pred, policy_target)
            value_loss = criterion_value(value_pred, value_target)
            total_loss = policy_loss + (20.0 * value_loss)

            total_loss.backward()
            optimizer.step()

            running_train_total += total_loss.item()
            running_train_policy += policy_loss.item()
            running_train_value += value_loss.item()
            
            with torch.no_grad():
                _, top_preds = policy_pred.topk(5, dim=1)
                target_expanded = policy_target.unsqueeze(1)
                running_train_top1 += (top_preds[:, :1] == target_expanded).sum().item()
                running_train_top3 += (top_preds[:, :3] == target_expanded).sum().item()
                running_train_top5 += (top_preds[:, :5] == target_expanded).sum().item()
            
        avg_train_total = running_train_total / len(train_dataloader)
        avg_train_policy = running_train_policy / len(train_dataloader)
        avg_train_value = running_train_value / len(train_dataloader)
        
        total_train_samples = len(train_dataloader.dataset)
        train_acc_top1 = (running_train_top1 / total_train_samples) * 100.0
        train_acc_top3 = (running_train_top3 / total_train_samples) * 100.0
        train_acc_top5 = (running_train_top5 / total_train_samples) * 100.0
        
        history_train_total.append(avg_train_total)
        history_train_policy.append(avg_train_policy)
        history_train_value.append(avg_train_value)
        history_epochs.append(epoch + 1)

        # --- VALIDATION PHASE ---
        avg_val_total, avg_val_pol, avg_val_val, val_top1, val_top3, val_top5 = evaluate_model(
            model, val_dataloader, criterion_policy, criterion_value, device
        )
        
        history_val_total.append(avg_val_total)
        history_val_policy.append(avg_val_pol)
        history_val_value.append(avg_val_val)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_total:.4f} (Pol: {avg_train_policy:.4f}, Val: {avg_train_value:.4f})")
        print(f"         | Train Acc  - Top-1: {train_acc_top1:.1f}%, Top-3: {train_acc_top3:.1f}%, Top-5: {train_acc_top5:.1f}%")
        print(f"Epoch {epoch+1} | Val Loss  : {avg_val_total:.4f} (Pol: {avg_val_pol:.4f}, Val: {avg_val_val:.4f})")
        print(f"         | Val Acc    - Top-1: {val_top1:.1f}%, Top-3: {val_top3:.1f}%, Top-5: {val_top5:.1f}%")

        # --- SCHEDULER & CHECKPOINTING ---
        scheduler.step(avg_val_total)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Current LR: {current_lr:.6f}")

        if avg_val_total < best_val_loss:
            print(f"--> Validation loss improved from {best_val_loss:.4f} to {avg_val_total:.4f}. Saving checkpoint!")
            best_val_loss = avg_val_total
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_total,
                },
                checkpoint_path
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"--> No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        # --- EARLY STOPPING ---
        if epochs_without_improvement >= patience:
            print(f"\n[!] Early stopping triggered after {epoch+1} epochs.")
            break
            
    # --- FINAL BLIND TEST EVALUATION ---
    print("\n=======================================================")
    print("TRAINING COMPLETE. LOADING BEST CHECKPOINT FOR FINAL TEST")
    print("=======================================================")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    test_total, test_pol, test_val, test_top1, test_top3, test_top5 = evaluate_model(
        model, test_dataloader, criterion_policy, criterion_value, device
    )
    
    print(f"FINAL BLIND TEST LOSS : {test_total:.4f} (Pol: {test_pol:.4f}, Val: {test_val:.4f})")
    print(f"FINAL BLIND TEST ACC  : Top-1: {test_top1:.1f}%, Top-3: {test_top3:.1f}%, Top-5: {test_top5:.1f}%")
        
    return (history_train_total, history_train_policy, history_train_value, 
            history_val_total, history_val_policy, history_val_value, history_epochs)

def plot_train_and_val(
    history_train_total: list[float], history_train_policy: list[float], history_train_value: list[float],
    history_val_total: list[float], history_val_policy: list[float], history_val_value: list[float],
    history_epochs: list[int]
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def format_subplot(ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=12)

    axes[0].plot(history_epochs, history_train_total, label='Train Total', color='blue', linewidth=2, marker='o')
    axes[0].plot(history_epochs, history_val_total, label='Val Total', color='orange', linewidth=2, linestyle='--', marker='s')
    axes[0].set_ylabel('Loss', fontsize=12)
    format_subplot(axes[0], 'Total Combined Loss')

    axes[1].plot(history_epochs, history_train_policy, label='Train Policy', color='green', linewidth=2, marker='o')
    axes[1].plot(history_epochs, history_val_policy, label='Val Policy', color='red', linewidth=2, linestyle='--', marker='s')
    format_subplot(axes[1], 'Policy Loss (Actor)')

    axes[2].plot(history_epochs, history_train_value, label='Train Value', color='purple', linewidth=2, marker='o')
    axes[2].plot(history_epochs, history_val_value, label='Val Value', color='brown', linewidth=2, linestyle='--', marker='s')
    format_subplot(axes[2], 'Value Loss (Critic)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = LosAlamosDataset("./data/datasets/los_alamos_dataset.h5")
    dataset_length = len(dataset)

    # generate seed for reproductibility
    generator = torch.Generator().manual_seed(42)

    training_len = int(0.8 * dataset_length)
    val_len = int(0.1 * dataset_length)
    test_len = dataset_length - training_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [training_len, val_len, test_len], generator=generator
    )

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    cnn = CNN_Residual_Dual_Head_network(num_residual_blocks=6, out_channel_conv=64).to(device)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.SmoothL1Loss() 
    
    optimizer = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4) # L2 regularization implemented
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # addaptive lr

    num_epochs = 120
    early_stopping_patience = 12 # Stop if validation fails to improve for 12 epochs

    history_train_total, history_train_policy, history_train_value, history_val_total, history_val_policy, history_val_value, history_epochs = train_dual_head_network(
        model=cnn, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        test_dataloader=test_dataloader, 
        criterion_policy=criterion_policy, 
        criterion_value=criterion_value, 
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs, 
        device=device,
        patience=early_stopping_patience
    )

    plot_train_and_val(
        history_train_total, history_train_policy, history_train_value, 
        history_val_total, history_val_policy, history_val_value, 
        history_epochs
    )