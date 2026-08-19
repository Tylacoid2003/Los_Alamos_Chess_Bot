import multiprocessing as mp
import h5py
import numpy as np
import chess
import os

from src.env.los_alamos_board import LosAlamosChess
from src.oracle.fairy_client import FairyClient
from src.env.board_translator import to_fairy_fen
from src.utils.move_conversion import to_standard_move
from src.oracle.random_openings import apply_random_opening

# ---------------------------------------------------------
# Core Game Logic
# ---------------------------------------------------------
def generate_game(client: FairyClient, random_plies: int = 8, engine_depth: int = 12) -> list:
    """
    Generates one self-play game using Fairy Stockfish.
    Each stored sample contains: [fairy_fen, raw_score, best_move, game_result]
    where raw_score is the absolute (White-perspective) centipawn score.
    """
    board = LosAlamosChess()
    apply_random_opening(board, random_plies)

    game_data = []

    while not board.is_game_over():
        current_fen = board.fen()
        fairy_fen = to_fairy_fen(current_fen)

        analysis = client.analyze(fairy_fen, engine_depth)

        best_move = analysis["best_move"]
        score_type = analysis["score_type"]
        score_value = analysis["score_value"]

        if best_move is None or best_move == "(none)" or score_value is None:
            break

        # Normalize mate scores to large numbers (White-perspective)
        if score_type == "mate":
            score_value = 10000 if score_value > 0 else -10000

        # Store raw absolute score (White's perspective) – DO NOT flip or scale
        raw_absolute_score = score_value

        # Translate Fairy move -> internal 8x8 move
        standard_move_str = to_standard_move(best_move)
        base_move_str = standard_move_str[:4]
        base_move_obj = chess.Move.from_uci(base_move_str)

        # Divergence check
        assert base_move_obj in board.legal_moves, (
            "\n"
            "==============================\n"
            "RULE DIVERGENCE DETECTED\n"
            "==============================\n"
            f"Fairy move      : {best_move}\n"
            f"Translated move : {standard_move_str}\n"
            f"Board FEN       : {current_fen}\n"
        )

        # Store (fairy_fen, raw_score, best_move)
        game_data.append([fairy_fen, raw_absolute_score, best_move])

        # Advance game
        board.push(standard_move_str)

    # Final game result (absolute, White-perspective)
    winner = board.winner()
    if winner is True:
        winner_value = 1
    elif winner is False:
        winner_value = -1
    else:
        winner_value = 0

    # Attach winner to every position
    for sample in game_data:
        sample.append(winner_value)

    return game_data

# ---------------------------------------------------------
# Worker Initialization (Runs once per CPU core)
# ---------------------------------------------------------
local_client = None

def init_worker(engine_path: str):
    """Initializes a persistent FairyClient for each separate process."""
    global local_client
    local_client = FairyClient(engine_path)
    local_client.connect()

    # Optional: limit engine resources to prevent oversubscription
    # local_client.set_option("Threads", 1)
    # local_client.set_option("Hash", 32)

def generate_game_worker(args):
    """Wrapper to call generate_game using the process's local client."""
    random_plies, engine_depth = args
    return generate_game(local_client, random_plies=random_plies, engine_depth=engine_depth)

# ---------------------------------------------------------
# Main Dataset Orchestrator (Parallel)
# ---------------------------------------------------------
def get_dataset_parallel(
    engine_path: str,
    num_games: int,
    output_path: str,
    random_plies: int = 8,
    engine_depth: int = 12
):
    all_fens = []
    all_values = []   # raw absolute scores (int32)
    all_moves = []
    all_winners = []

    cpu_count = mp.cpu_count()
    print(f"Generating {num_games} games across {cpu_count} CPU cores...")
    print(f"  - Random plies: {random_plies}")
    print(f"  - Engine depth: {engine_depth}\n")

    # Spin up a pool of workers
    with mp.Pool(
        processes=cpu_count,
        initializer=init_worker,
        initargs=(engine_path,)
    ) as pool:
        tasks = [(random_plies, engine_depth)] * num_games

        completed_games = 0
        for game_data in pool.imap_unordered(generate_game_worker, tasks):
            for fen, value, move, winner in game_data:
                all_fens.append(fen)
                all_values.append(value)        # raw int
                all_moves.append(move)
                all_winners.append(winner)

            completed_games += 1
            if completed_games % 100 == 0:
                print(f"Completed {completed_games}/{num_games} games.")

    print("\nSaving dataset...\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset("fens", data=np.asarray(all_fens, dtype="S"))
        # Store raw scores as int32 (Stockfish centipawns)
        h5file.create_dataset("values", data=np.asarray(all_values, dtype=np.int32))
        h5file.create_dataset("moves", data=np.asarray(all_moves, dtype="S"))
        h5file.create_dataset("winners", data=np.asarray(all_winners, dtype=np.int8))

    print(f"Dataset successfully saved to {output_path}")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    engine_path = "./fairy-stockfish.exe"   # adjust path as needed
    output_path = "./data/datasets/los_alamos_dataset2.h5"

    get_dataset_parallel(
        engine_path=engine_path,
        num_games=50000,
        output_path=output_path,
        random_plies=8,      # matches RL exploration cutoff
        engine_depth=14      # optimal speed/quality trade-off
    )