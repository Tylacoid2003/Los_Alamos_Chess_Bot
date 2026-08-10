import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_data(filepath):
    """Utility to load columns from a CSV into a dictionary of lists."""
    if not os.path.exists(filepath):
        return None
    
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for h in headers:
            data[h] = []
            
        for row in reader:
            for i, val in enumerate(row):
                # Convert to float if possible, else keep as string
                try:
                    data[headers[i]].append(float(val))
                except ValueError:
                    data[headers[i]].append(val)
    return data

def moving_average(data, window_size=50):
    """Smooths noisy RL data for better visualization."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_dashboard(train_csv_path, eval_csv_path):
    train_data = load_csv_data(train_csv_path)
    eval_data = load_csv_data(eval_csv_path)

    if not train_data:
        print(f"Error: Could not find {train_csv_path}")
        return

    # Create a dashboard with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    window = min(50, len(train_data["Episode"]) // 5) # Adaptive smoothing window

    # --- Plot 1: Episode Length (Moving Average) ---
    episodes_smoothed = train_data["Episode"][(window-1):]
    length_smoothed = moving_average(train_data["Length"], window)
    
    axes[0].plot(train_data["Episode"], train_data["Length"], alpha=0.2, color='gray', label='Raw Length')
    axes[0].plot(episodes_smoothed, length_smoothed, color='blue', linewidth=2, label=f'MA ({window})')
    axes[0].set_title("Game Length over Time", fontweight='bold')
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Number of Moves")
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].legend()

    # --- Plot 2: Total Loss (Moving Average) ---
    loss_smoothed = moving_average(train_data["Total_Loss"], window)
    axes[1].plot(train_data["Episode"], train_data["Total_Loss"], alpha=0.2, color='gray', label='Raw Loss')
    axes[1].plot(episodes_smoothed, loss_smoothed, color='red', linewidth=2, label=f'MA ({window})')
    axes[1].set_title("RL Total Loss", fontweight='bold')
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].legend()

    # --- Plot 3: Arena Evaluation Win Rates ---
    if eval_data and len(eval_data["Episode"]) > 0:
        episodes = eval_data["Episode"]
        rl_wins = np.array(eval_data["RL_Wins"])
        base_wins = np.array(eval_data["Baseline_Wins"])
        draws = np.array(eval_data["Draws"])
        
        # Calculate percentages
        total_games = rl_wins + base_wins + draws
        rl_pct = (rl_wins / total_games) * 100
        base_pct = (base_wins / total_games) * 100
        draw_pct = (draws / total_games) * 100

        # Stacked bar chart for win rates
        axes[2].bar(episodes, rl_pct, width=20, label='RL Agent Wins', color='green')
        axes[2].bar(episodes, draw_pct, width=20, bottom=rl_pct, label='Draws', color='gray')
        axes[2].bar(episodes, base_pct, width=20, bottom=rl_pct+draw_pct, label='Baseline Wins', color='red')
        
        axes[2].set_title("Arena Evaluation vs Pre-Trained", fontweight='bold')
        axes[2].set_xlabel("Training Episode")
        axes[2].set_ylabel("Win Rate (%)")
        axes[2].legend(loc='lower left')
    else:
        axes[2].text(0.5, 0.5, 'No Evaluation Data Yet', horizontalalignment='center', verticalalignment='center')
        axes[2].set_title("Arena Evaluation vs Pre-Trained", fontweight='bold')

    plt.tight_layout()
    
    # Save and display
    save_dir = "./reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "rl_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "rl_dashboard.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_log = "./reports/rl_training_log.csv"
    eval_log = "./reports/rl_evaluation_log.csv"
    plot_dashboard(train_log, eval_log)