import unittest
import numpy as np
from TSPsolver_fd9.algorithms.random_search import (
    random_search,
    calculate_distance,
    generate_distance_matrix,
    get_nearest_neighbor,
)

class TestRandomSearch(unittest.TestCase):
    def test_calculate_distance(self):
        # Check Euclidean distance between known points.
        self.assertAlmostEqual(calculate_distance((0, 0), (3, 4)), 5.0, places=5)
        self.assertAlmostEqual(calculate_distance((1, 1), (4, 5)), 5.0, places=5)

    def test_generate_distance_matrix(self):
        # Create a simple set of 3 cities.
        cities = np.array([
            [0, 0],
            [3, 4],
            [6, 8]
        ])
        D = generate_distance_matrix(cities)
        self.assertEqual(D.shape, (3, 3))
        self.assertAlmostEqual(D[0, 1], 5.0, places=5)
        self.assertAlmostEqual(D[1, 2], 5.0, places=5)
        self.assertAlmostEqual(D[0, 2], 10.0, places=5)

    def test_get_nearest_neighbor(self):
        # Given 3 cities, for city 0 the nearest (if only city 0 is visited) should be city 1.
        cities = np.array([
            [0, 0],
            [3, 4],
            [6, 8]
        ])
        D = generate_distance_matrix(cities)
        dist, nn = get_nearest_neighbor(0, D, {0})
        self.assertEqual(nn, 1)
        self.assertAlmostEqual(dist, 5.0, places=5)

    def test_random_search(self):
        # Test random_search returns a valid tour covering all cities.
        cities = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ])
        path, total_cost = random_search(cities)
        # Verify the path includes each city exactly once.
        self.assertEqual(len(path), len(cities))
        self.assertEqual(set(path), set(range(len(cities))))
        self.assertGreater(total_cost, 0)

if __name__ == '__main__':
    unittest.main()
