import unittest
import numpy as np
from TSPsolver_fed.algorithms.hill_climbing import hill_climbing, calculate_distance

class TestHillClimbing(unittest.TestCase):
    def test_hill_climbing_route(self):
        # Test on a simple square (4 cities)
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        route, cost = hill_climbing(points)
        self.assertEqual(len(route), len(points))
        # Check that route contains all cities
        self.assertEqual(set(route), set(range(len(points))))
        self.assertGreater(cost, 0)

    def test_calculate_distance(self):
        # Test calculate_distance on a known route.
        points = np.array([
            [0, 0],
            [3, 0],
            [3, 4]
        ])
        # Route: 0 -> 1 -> 2 -> 0: distances: 3, 4, 5; total = 12.
        dist = calculate_distance(points)
        self.assertAlmostEqual(dist, 12, places=1)

if __name__ == '__main__':
    unittest.main()
