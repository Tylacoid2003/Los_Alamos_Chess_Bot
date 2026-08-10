def to_fairy_fen(standard_fen: str) -> str:
    """
    Converts a standard FEN string to a fairy FEN string for Los Alamos Chess.
    
    Args:
        standard_fen (str): The standard FEN string to convert.
    """
    fen_parts = standard_fen.split(' ')
    board_part = fen_parts[0]
    turn_part = fen_parts[1]
    castling_part = fen_parts[2]
    en_passant_part = fen_parts[3]
    halfmove_clock_part = fen_parts[4]
    fullmove_number_part = fen_parts[5]

    rows = board_part.split('/')
    extended_grid = []
    for row in rows:
        extended_row = ""
        for char in row:
            if char.isdigit():
                extended_row += ' ' * int(char)  # Replace digit with that many empty squares
            else:
                extended_row += char
        extended_grid.append(extended_row)

    grid_6x6 = [row[1:7] for row in extended_grid[1:7]]  # Extract the 6x6 grid from the extended grid

    # Convert the 6x6 grid back to FEN format
    fairy_fen_rows = []
    for row in grid_6x6:
        count = 0
        fen_row = ""
        for char in row:
            if char == ' ':
                count += 1
            else:
                if count > 0:
                    fen_row += str(count)
                    count = 0
                fen_row += char
        if count > 0:
            fen_row += str(count)
        fairy_fen_rows.append(fen_row)

    fairy_fen_board = '/'.join(fairy_fen_rows)
    fairy_fen = fairy_fen_board + " " + turn_part + " " + castling_part + " " + en_passant_part + " " + halfmove_clock_part + " " + fullmove_number_part
    return fairy_fen 

