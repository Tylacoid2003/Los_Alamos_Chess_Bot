import unittest
import chess
from src.env.los_alamos_board import LosAlamosChess 

class TestLosAlamosBoard(unittest.TestCase):
    def setUp(self):
        """Runs before every test to give us a fresh board."""
        self.game = LosAlamosChess()

    def test_initialization(self):
        """Test that the board initializes correctly and pawns can't double jump."""
        # Using self.game.board.fen() since get_fen() is not in your class
        self.assertEqual(self.game.board.fen(), self.game.init_FEN)
        
        # Ensure legal moves are generated and none go out of bounds.
        for move in self.game.legal_moves:
            self.assertIn(move.from_square, self.game.VALID_SQUARES)
            self.assertIn(move.to_square, self.game.VALID_SQUARES)

    def test_illegal_move_rejection(self):
        """Test that pushing a move into the padding raises an error."""
        # b2 to a1 attempts to move a piece off the 6x6 grid into the padding
        illegal_move = chess.Move.from_uci("b2a1")
        
        with self.assertRaises(ValueError):
            self.game.push(illegal_move)

    def test_push_and_pop(self):
        """Test that we can make a move and undo it."""
        initial_fen = self.game.board.fen()
        
        # Pick the first legal move available
        first_move = self.game.legal_moves[0]
        
        self.game.push(first_move)
        self.assertNotEqual(self.game.board.fen(), initial_fen)
        
        self.game.pop()
        self.assertEqual(self.game.board.fen(), initial_fen)

    def test_custom_checkmate_against_wall(self):
        """
        Tests the edge case where the King is mated against the 6x6 boundary.
        Standard chess thinks the King can escape into the 8x8 padding, 
        so we must prove our custom logic catches the checkmate.
        """
        # FEN Setup: White King on b2. Black Queen on b3. Black King on c4.
        # Inside the 6x6 grid, the White King is in check and has no safe squares.
        mate_fen = "8/8/8/8/2k5/1q6/1K6/8 w - - 0 1"
        self.game.board.set_fen(mate_fen)
        
        # 1. Prove standard chess thinks the game is STILL GOING
        self.assertFalse(self.game.board.is_game_over())
        
        # 2. Prove our custom logic correctly identifies the GAME IS OVER
        self.assertTrue(self.game.is_game_over())
        
        # 3. Prove Black is declared the winner using your custom winner() method
        self.assertEqual(self.game.winner(), chess.BLACK)

if __name__ == '__main__':
    unittest.main()