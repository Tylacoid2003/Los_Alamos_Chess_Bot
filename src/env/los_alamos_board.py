import chess

class LosAlamosChess:
    # Precompute valid squares once at the class level
    VALID_SQUARES = {
        square for square in chess.SQUARES
        if 1 <= chess.square_file(square) <= 6 and 1 <= chess.square_rank(square) <= 6
    }
    INIT_FEN = "8/1rnqknr1/1pppppp1/8/8/1PPPPPP1/1RNQKNR1/8 w - - 0 1"

    def __init__(self, board=None):
        if board is not None:
            self.board = board
        else:
            self.board = chess.Board(self.INIT_FEN)

    def copy(self):
        """Fast shallow clone avoiding heavy python-chess stack copies and deepcopy overhead."""
        new_obj = LosAlamosChess.__new__(LosAlamosChess)
        new_obj.board = self.board.copy(stack=False)
        return new_obj

    def invisible_wall(self, standard_legal_moves):
        return [
            move for move in standard_legal_moves
            if move.from_square in self.VALID_SQUARES and move.to_square in self.VALID_SQUARES
        ]

    @property
    def legal_moves(self):
        standard_legal_moves = list(self.board.legal_moves)
        moves_in_bounds = self.invisible_wall(standard_legal_moves)
        
        valid_moves = []
        for move in moves_in_bounds:
            # Block 8x8 pawn double pushes
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                if abs(from_rank - to_rank) == 2:
                    continue
            valid_moves.append(move)
            
        return valid_moves

    def push(self, move_str: str, validate: bool = False):
        """
        Pushes a move to the board. 
        Setting validate=False skips re-generating the legal move list (huge speedup during MCTS).
        """
        if isinstance(move_str, chess.Move):
            move_str = move_str.uci()
            
        base_move = move_str[:4]
        promo_char = move_str[4] if len(move_str) > 4 else None
        standard_move = chess.Move.from_uci(base_move)
        
        if validate and standard_move not in self.legal_moves:
            raise ValueError(f"Illegal move {move_str} attempted in Los Alamos Chess.")

        self.board.push(standard_move)
        
        if promo_char:
            dest_square = chess.parse_square(base_move[2:4])
            piece = self.board.piece_at(dest_square)
            if piece:
                symbol = promo_char.upper() if piece.color == chess.WHITE else promo_char.lower()
                self.board.set_piece_at(dest_square, chess.Piece.from_symbol(symbol))

    def is_game_over(self):
        if self.board.is_game_over(claim_draw=True):
            return True
        return len(self.legal_moves) == 0

    def reset(self):
        self.board = chess.Board(self.INIT_FEN)

    def pop(self):
        self.board.pop()

    def winner(self):
        standard_outcome = self.board.outcome()
        if standard_outcome is not None:
            return standard_outcome.winner

        if len(self.legal_moves) == 0:
            if self.board.is_check():
                return not self.board.turn
            else:
                return None  # Stalemate

        return None

    def fen(self) -> str:
        return self.board.fen()