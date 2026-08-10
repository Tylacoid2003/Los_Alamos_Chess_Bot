import unittest
import os
from src.oracle.fairy_client import FairyClient

class TestFairyClient(unittest.TestCase):
    def setUp(self):
        # UPDATE THIS PATH to wherever your Fairy-Stockfish executable is located!
        self.engine_path = "./fairy-stockfish"
        
        # We only run the tests if the engine actually exists there
        if not os.path.exists(self.engine_path):
            self.skipTest(f"Engine not found at {self.engine_path}. Update the path!")
            
        self.client = FairyClient(self.engine_path)
        self.client.connect()

    def tearDown(self):
        """This runs after the test finishes to guarantee the engine is closed."""
        if hasattr(self, 'client'):
            self.client.quit()

    def test_engine_analysis(self):
        """Test if the engine can analyze the Los Alamos starting position."""
        # The native 6x6 starting position for Los Alamos chess
        start_fen = "rnqknr/pppppp/6/6/PPPPPP/RNQKNR w - - 0 1"
        
        # Ask the engine to think up to depth 5 (should be very fast)
        result = self.client.analyze(start_fen, depth=5)
        
        # Verify it returns a dictionary with the expected keys
        self.assertIn("best_move", result)
        self.assertIn("score_value", result)
        
        # Verify the move looks like a 6x6 chess move (e.g., 'b2b3' or 'g1f3' natively 'b1c3' etc)
        # It should be a string of at least 4 characters
        self.assertIsInstance(result["best_move"], str)
        self.assertGreaterEqual(len(result["best_move"]), 4)
        
        # Verify the score is an integer
        self.assertIsInstance(result["score_value"], int)
        
        print(f"\nSuccess! Engine suggests: {result['best_move']} with score {result['score_value']}")

if __name__ == '__main__':
    unittest.main()