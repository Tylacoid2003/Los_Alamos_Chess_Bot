import h5py
import numpy as np
import chess

from src.env.los_alamos_board import LosAlamosChess
from src.oracle.fairy_client import FairyClient
from src.env.board_translator import to_fairy_fen
from src.utils.move_conversion import to_standard_move
from src.oracle.random_openings import apply_random_opening


def generate_game(client: FairyClient, random_plies: int = 4, engine_depth: int = 10) -> list:
    """
    Generates one self-play game using Fairy Stockfish.

    Each stored sample contains

        [fairy_fen, value_target, best_move, game_result]
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

        if (best_move is None or best_move == "(none)" or score_value is None):
            break

        #
        # Normalize mate scores into the same numerical space
        #
        if score_type == "mate":
            score_value = 10000 if score_value > 0 else -10000

        #
        # Translate Fairy move -> internal 8x8 move
        #
        standard_move_str = to_standard_move(best_move)
        
        # Extract only the physical coordinates (e.g., "f3e2")
        base_move_str = standard_move_str[:4]
        base_move_obj = chess.Move.from_uci(base_move_str)

        #
        # Divergence tripwire (check only the base physics!)
        #
        assert base_move_obj in board.legal_moves, (
            "\n"
            "==============================\n"
            "RULE DIVERGENCE DETECTED\n"
            "==============================\n"
            f"Fairy move      : {best_move}\n"
            f"Translated move : {standard_move_str}\n"
            f"Board FEN       : {current_fen}\n"
        )

        #
        # Store position
        #
        game_data.append([fairy_fen, score_value, best_move])

        #
        # Advance game (pass the FULL string so your environment handles the promotion!)
        #
        board.push(standard_move_str)

    #
    # Final game result
    #
    winner = board.winner()

    if winner is True:
        winner_value = 1
    elif winner is False:
        winner_value = -1
    else:
        winner_value = 0

    #
    # Attach winner to every position
    #
    for sample in game_data:
        sample.append(winner_value)

    return game_data


def get_dataset(client: FairyClient, num_games: int, output_path: str, engine_depth: int = 10):
    """
    Generates an imitation-learning dataset.

    Stored fields

        fens
        values
        moves
        winners
    """

    all_fens = []
    all_values = []
    all_moves = []
    all_winners = []

    print(f"Generating {num_games} games...\n")

    for game_index in range(num_games):
        game = generate_game(client, random_plies=4, engine_depth=engine_depth)

        for fen, value, move, winner in game:
            all_fens.append(fen)
            all_values.append(value)
            all_moves.append(move)
            all_winners.append(winner)

        if (game_index + 1) % 10 == 0:
            print(f"Completed {game_index + 1}/{num_games} games.")

    print("\nSaving dataset...\n")

    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset("fens", data=np.asarray(all_fens, dtype="S"))

        h5file.create_dataset("values", data=np.asarray(all_values, dtype=np.int32))

        h5file.create_dataset("moves", data=np.asarray(all_moves, dtype="S"))

        h5file.create_dataset("winners", data=np.asarray(all_winners, dtype=np.int8))

    print("Dataset successfully saved.")


if __name__ == "__main__":

    engine_path = "./fairy-stockfish"
    output_path = "./data/datasets/los_alamos_dataset.h5"

    client = FairyClient(engine_path)
    client.connect()

    try:
        get_dataset(client=client, num_games=10000, output_path=output_path, engine_depth=10)

    finally:
        client.quit()