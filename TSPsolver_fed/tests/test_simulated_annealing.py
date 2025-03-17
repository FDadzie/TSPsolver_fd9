import math
import random
import matplotlib.pyplot as plt

# Calculate the Euclidean distance between two points.
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)]

# Compute the total distance of the tour (including return to start).
def total_distance(tour, points):
    d = 0.0
    num_points = len(tour)
    for i in range(num_points):
        d += distance(points[tour[i]], points[tour[(i + 1) % num_points]])
    return d

def simulated_annealing(points, initial_temperature=10000, cooling_rate=0.995,
                        stopping_temperature=1e-8, iterations_per_temp=100):
    """
    Solve the TSP using simulated annealing.
    
    Parameters:
        points: List of (x, y) coordinates.
        initial_temperature: Starting temperature.
        cooling_rate: Factor to reduce the temperature at each step.
        stopping_temperature: Temperature at which the algorithm stops.
        iterations_per_temp: Number of iterations to perform at each temperature.
    
    Returns:
        best_tour: The best found tour as a list of point indices.
        best_distance: The total distance of the best tour.
    """
    n = len(points)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_distance = total_distance(current_tour, points)
    
    best_tour = current_tour[:]
    best_distance = current_distance
    T = initial_temperature
    
    while T > stopping_temperature:
        for _ in range(iterations_per_temp):
            # Create a new candidate by swapping two cities
            new_tour = current_tour[:]
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            new_distance = total_distance(new_tour, points)
            delta = new_distance - current_distance
            
            # Accept the new tour if it's better, or with a probability if it's worse
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_tour = new_tour
                current_distance = new_distance
                if current_distance < best_distance:
                    best_tour = current_tour[:]
                    best_distance = current_distance
        T *= cooling_rate  # Cool down
    return best_tour, best_distance

if __name__ == '__main__':
    # Example: generate some random points
    num_points = 20
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
    
    best_tour, best_distance = simulated_annealing(points)
    
    print("Best tour:", best_tour)
    print("Best distance:", best_distance)
    
    # Plot the resulting tour
    tour_points = [points[i] for i in best_tour] + [points[best_tour[0]]]
    xs, ys = zip(*tour_points)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('TSP Tour using Simulated Annealing')
    plt.show()