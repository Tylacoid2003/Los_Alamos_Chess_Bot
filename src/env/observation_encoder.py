import numpy as np

piece_to_channel = {
    "P": 0, "R": 1, "N": 2, "Q": 3, "K": 4,  # White pieces
    "p": 5, "r": 6, "n": 7, "q": 8, "k": 9   # Black pieces
}

def encode_fen(fairy_fen: str) -> np.ndarray:
    state_space = np.zeros(shape=(11,6,6), dtype=np.float32)

    fen_splited = fairy_fen.split()
    board_layout = fen_splited[0]
    turn_indicator = fen_splited[1] if len(fen_splited) > 1 else 'w'

    board_rows = board_layout.split("/")
    for rank_idx, row in enumerate(board_rows):
        file_idx = 0
        for _, piece in enumerate(row):
            if piece.isdigit():
                file_idx += int(piece)
            elif piece in piece_to_channel:
                channel = piece_to_channel[piece]
                state_space[channel, rank_idx, file_idx] = 1.0
                file_idx += 1
    if turn_indicator == 'w':
        state_space[10, :, :] = 1.0

    return state_space

def build_action_space() -> dict:

    files = ['a', 'b', 'c', 'd', 'e', 'f']
    ranks = ['1', '2', '3', '4', '5', '6']

    board_positions = [f"{file}{rank}" for file in files for rank in ranks]

    base_moves = [f"{from_sq}{to_sq}" for from_sq in board_positions for to_sq in board_positions if from_sq != to_sq]

    all_moves = []
    promotion_chars = ['q', 'r', 'n']

    for move in base_moves:
        all_moves.append(move)

        from_file, from_rank = move[0], move[1]
        to_file, to_rank = move[2], move[3]

        is_white_promo = (from_rank == '5' and to_rank == '6')
        is_black_promo = (from_rank == '2' and to_rank == '1')

        file_diff = abs(ord(to_file) - ord(from_file))
        is_valid_geo = file_diff <= 1

        if (is_white_promo or is_black_promo) and is_valid_geo:
            for piece in promotion_chars:
                prom_move = move+piece
                all_moves.append(prom_move)

    action_space = {f"{move}": idx for idx, move in enumerate(all_moves)}

    return action_space