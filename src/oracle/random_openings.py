import random

def apply_random_opening(board, num_plies: int = 4):
    """
    Plays a sequence of random legal moves to create diverse starting 
    positions for dataset generation.
    
    Args:
        board: The LosAlamosChess environment instance.
        num_plies (int): The number of random half-moves to play.
                         Default is 4 (2 full turns for each player).
    """
    for _ in range(num_plies):
        # board.legal_moves generates a dynamic object, so we cast it to a list
        legal_moves = list(board.legal_moves)
        
        # Safety net: If the game somehow ends during the random opening, stop early
        if not legal_moves:
            break
            
        random_move = random.choice(legal_moves)
        board.push(random_move)