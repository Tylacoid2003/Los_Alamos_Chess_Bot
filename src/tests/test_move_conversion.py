import unittest
from src.utils.move_conversion import to_fairy_move, to_standard_move

class TestMoveConversion(unittest.TestCase):
    def test_to_fairy_move(self):
        """Test converting from padded 8x8 standard moves to 6x6 fairy moves."""
        # Normal move: b2 to b3 -> a1 to a2
        self.assertEqual(to_fairy_move("b2b3"), "a1a2")
        
        # Corner move: g7 to f6 -> f6 to e5
        self.assertEqual(to_fairy_move("g7f6"), "f6e5")
        
        # Promotion move: c6 to c7 with a Queen promotion
        self.assertEqual(to_fairy_move("c6c7q"), "b5b6q")

    def test_to_standard_move(self):
        """Test converting from 6x6 fairy moves to padded 8x8 standard moves."""
        # Normal move: a1 to a2 -> b2 to b3
        self.assertEqual(to_standard_move("a1a2"), "b2b3")
        
        # Corner move: f6 to e5 -> g7 to f6
        self.assertEqual(to_standard_move("f6e5"), "g7f6")
        
        # Promotion move: d5 to d6 with a Rook promotion
        self.assertEqual(to_standard_move("d5d6r"), "e6e7r")

    def test_reversibility(self):
        """Test that converting back and forth returns the original string."""
        # Test standard move reversibility
        original_move = "e2e4"
        fairy = to_fairy_move(original_move)
        standard = to_standard_move(fairy)
        self.assertEqual(original_move, standard)

        # Test promotion move reversibility
        prom_move = "b6a7n"
        fairy_prom = to_fairy_move(prom_move)
        standard_prom = to_standard_move(fairy_prom)
        self.assertEqual(prom_move, standard_prom)

if __name__ == '__main__':
    unittest.main()