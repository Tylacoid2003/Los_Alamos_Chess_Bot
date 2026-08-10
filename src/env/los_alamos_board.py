import chess

class LosAlamosChess():
    def __init__(self):
        self.init_FEN = "8/1rnqknr1/1pppppp1/8/8/1PPPPPP1/1RNQKNR1/8 w - - 0 1"
        self.board = chess.Board(self.init_FEN)
        self.VALID_SQUARES = {
            square for square in chess.SQUARES
                if 1 <= chess.square_file(square) <= 6 and
                   1 <= chess.square_rank(square) <= 6
        }

    def invisible_wall(self, standard_legal_moves):
        return [
            move for move in standard_legal_moves
            if move.from_square in self.VALID_SQUARES and move.to_square in self.VALID_SQUARES
        ]

    @property
    def legal_moves(self):
        standard_legal_moves = list(self.board.legal_moves)

        # Filter out moves that go outside the 6x6 board area
        moves_in_bounds = self.invisible_wall(standard_legal_moves)
        
        valid_moves = []
        for move in moves_in_bounds:
            # Block 8x8 pawn double pushes
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                # If the pawn jumped 2 ranks, discard the move
                if abs(from_rank - to_rank) == 2:
                    continue
                    
            valid_moves.append(move)
            
        return valid_moves

    def push(self, move_str: str):
        # We accept a string (like "g6g7r" or "b2b3") to handle promotions cleanly
        if isinstance(move_str, chess.Move):
            move_str = move_str.uci()
            
        base_move = move_str[:4]
        promo_char = move_str[4] if len(move_str) > 4 else None
        
        standard_move = chess.Move.from_uci(base_move)
        
        # Check if the base movement is legal in Los Alamos Chess
        if standard_move in self.legal_moves:
            self.board.push(standard_move)
            
            # If it was a promotion, manually swap the piece in the internal board
            if promo_char:
                dest_square = chess.parse_square(base_move[2:4])
                piece_color = self.board.piece_at(dest_square).color
                symbol = promo_char.upper() if piece_color == chess.WHITE else promo_char.lower()
                self.board.set_piece_at(dest_square, chess.Piece.from_symbol(symbol))
        else:
            raise ValueError(f"Illegal move {move_str} attempted in Los Alamos Chess.")

    def is_game_over(self):
        # Check if the game is over either by standard chess rules or by having no legal moves in Los Alamos Chess
        if self.board.is_game_over() or len(self.legal_moves) == 0:
            return True
        return False

    def reset(self):
        self.board = chess.Board(self.init_FEN)

    def pop(self):
        # Go back one move in the game history
        self.board.pop()

    def winner(self):

        # First check if regular outcome is available
        standard_outcome = self.board.outcome()
        if standard_outcome is not None:
            return standard_outcome.winner # Returns True (White), False (Black), or None (Draw)

        # Check if based on the length of self.legal_moves, the game is over due to no legal moves in Los Alamos Chess
        if len(self.legal_moves) == 0:
            if self.board.is_check():
                return not self.board.turn  # If current player is in check and has no legal moves, opposite player wins
            else:
                return None  # Stalemate situation, no winner

        return None  # Game is not over, no winner yet

    def fen(self) -> str:
        """Returns the FEN string of the current board state."""
        return self.board.fen()
    