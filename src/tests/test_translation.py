import unittest
from src.env.board_translator import to_fairy_fen

class TestBoardTranslator(unittest.TestCase):
    def test_start_position(self):
        """Test that the initial 8x8 padded FEN converts to the standard 6x6 start FEN."""
        padded_start_fen = "8/1rnqknr1/1pppppp1/8/8/1PPPPPP1/1RNQKNR1/8 w - - 0 1"
        expected_fairy_fen = "rnqknr/pppppp/6/6/PPPPPP/RNQKNR w - - 0 1"
        
        result = to_fairy_fen(padded_start_fen)
        self.assertEqual(result, expected_fairy_fen)

    def test_mid_game_position(self):
        """Test a mid-game position with spaces scattered across the board."""
        # 8x8 Breakdown:
        # Row 1: 1 r 2 k 2 1 (8 squares) -> slices to -> r 2 k 2 (6 squares)
        # Row 2: 1 p 6       (8 squares) -> slices to -> p 5      (6 squares)
        # Row 5: 6 P 1       (8 squares) -> slices to -> 5 P      (6 squares)
        padded_mid_fen = "8/1r2k21/1p6/8/8/6P1/1R2K21/8 b - - 0 10"
        expected_fairy_fen = "r2k2/p5/6/6/5P/R2K2 b - - 0 10"
        
        result = to_fairy_fen(padded_mid_fen)
        self.assertEqual(result, expected_fairy_fen)

if __name__ == '__main__':
    unittest.main()