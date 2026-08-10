
def to_fairy_move(standard_move: str) -> str:
    """
    Converts a standard chess move notation to Los Alamos chess notation.
    
    Args:
        standard_move (str): The standard chess move notation (e.g. 'b2b3').
    
    Returns:
        str: The Los Alamos chess notation. (e.g., 'a1a2').
    """
    new_start_file = chr(ord(standard_move[0]) - 1)
    new_start_rank = str(int(standard_move[1])- 1)

    new_end_file = chr(ord(standard_move[2]) - 1)
    new_end_rank = str(int(standard_move[3]) - 1)

    # Capture the promotion piece (e.g., 'q') if it exists
    promotion = standard_move[4:] if len(standard_move) > 4 else ""

    return  new_start_file + new_start_rank + new_end_file + new_end_rank + promotion

def to_standard_move(fairy_move: str) -> str:
    """
    Converts a Los Alamos chess move notation to standard chess notation.
    
    Args:
        fairy_move (str): The Los Alamos chess move notation (e.g. 'a1a2').
    
    Returns:
        str: The standard chess notation. (e.g., 'b2b3').
    """
    new_start_file = chr(ord(fairy_move[0]) + 1)
    new_start_rank = str(int(fairy_move[1]) + 1)

    new_end_file = chr(ord(fairy_move[2]) + 1)
    new_end_rank = str(int(fairy_move[3]) + 1)

    # Capture the promotion piece (e.g., 'q') if it exists
    promotion = fairy_move[4:] if len(fairy_move) > 4 else ""

    return new_start_file + new_start_rank + new_end_file + new_end_rank + promotion