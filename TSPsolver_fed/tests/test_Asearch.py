import unittest
import numpy as np
from TSPsolver_fed.algorithms.Asearch import astar_tsp
from TSPsolver_fed.algorithms.simulated_annealing import compute_distance_matrix

class TestAStarTSP(unittest.TestCase):
    def test_astar_tsp_valid_route(self):
        # Use a small square instance (4 cities) with known structure.
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        D = compute_distance_matrix(points)
        route, cost = astar_tsp(D, start=0)
        # The route should be cyclic (start appended at the end)
        self.assertEqual(len(route), len(points) + 1)
        # All cities (except the duplicate start at end) must be visited.
        self.assertEqual(set(route[:-1]), set(range(len(points))))
        self.assertGreater(cost, 0)

if __name__ == '__main__':
    unittest.main()
