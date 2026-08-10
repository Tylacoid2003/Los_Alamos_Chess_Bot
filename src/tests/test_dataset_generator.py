import unittest
import os
from src.oracle.fairy_client import FairyClient
from src.oracle.dataset_generator import generate_game

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        # UPDATE THIS PATH just like you did in test_engine.py
        self.engine_path = "./fairy-stockfish" 
        
        if not os.path.exists(self.engine_path):
            self.skipTest(f"Engine not found at {self.engine_path}")
            
        self.client = FairyClient(self.engine_path)
        self.client.connect()

    def tearDown(self):
        if hasattr(self, 'client'):
            self.client.quit()

    def test_generate_single_game(self):
        """Test that the generator can successfully play a full game and record data."""
        # Run a game with very shallow depth so the test is fast
        game_data = generate_game(self.client, random_plies=4, engine_depth=3)
        
        # 1. Ensure the game actually produced data
        self.assertGreater(len(game_data), 0, "Game data should not be empty")
        
        # 2. Check the format of the first recorded turn
        first_turn = game_data[0]
        self.assertEqual(len(first_turn), 3, "Each data point should have 3 elements")
        
        fen, score, move = first_turn
        self.assertIsInstance(fen, str, "FEN should be a string")
        self.assertIsInstance(score, int, "Score should be an integer")
        self.assertIsInstance(move, str, "Move should be a string")
        
        print(f"\nSuccess! Generated a game with {len(game_data)} plies (turns).")

if __name__ == '__main__':
    unittest.main()