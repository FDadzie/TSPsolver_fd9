import unittest
import numpy as np
from TSPsolver_fed.algorithms.simulated_annealing import simulated_annealing, compute_distance_matrix, total_distance

class TestSimulatedAnnealing(unittest.TestCase):
    def test_simulated_annealing_square(self):
        # Define a square: (0,0), (0,1), (1,1), (1,0)
        points = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ])
        D = compute_distance_matrix(points)
        # Use parameters tuned for a small instance.
        best_tour, best_distance, iteration_count = simulated_annealing(
            D,
            initial_temperature=1000,
            cooling_rate=0.99,
            stopping_temperature=1e-3,
            iterations_per_temp=50
        )
        
        # Check that best_tour is a permutation of {0, 1, 2, 3}.
        self.assertEqual(set(best_tour), set(range(4)))
        
        # Verify that best_distance equals the tour's total distance.
        computed_distance = total_distance(best_tour, D)
        self.assertAlmostEqual(best_distance, computed_distance, places=4)
        
        # Ensure iteration_count is a positive integer.
        self.assertIsInstance(iteration_count, int)
        self.assertGreater(iteration_count, 0)

    def test_simulated_annealing_consistency(self):
        # Use another small instance and run multiple times.
        points = np.array([
            [0, 0],
            [0, 2],
            [2, 2],
            [2, 0]
        ])
        D = compute_distance_matrix(points)
        for _ in range(5):
            best_tour, best_distance, iteration_count = simulated_annealing(
                D,
                initial_temperature=1000,
                cooling_rate=0.99,
                stopping_temperature=1e-3,
                iterations_per_temp=50
            )
            self.assertEqual(set(best_tour), set(range(4)))
            self.assertGreater(best_distance, 0)
            self.assertGreater(iteration_count, 0)

if __name__ == '__main__':
    unittest.main()
