from graph import Graph
from qubo_problem import QAOA_TSP_Maxcut
from plot_solutions import draw_tsp_solution, sample_and_plot_histogram, interpret_solution
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from config import Config
import numpy as np


def main():
    # Initial configurations
    config = Config()

    # Example usage
    N = 5
    weight_range = (10, 100)
    seed = 51 

    # Instantiate and create the graph
    graph = Graph(N, weight_range, seed)
    graph.draw()

    # Generate the adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=int)
    for (u, v, data) in graph.G.edges(data=True):
        adj_matrix[u][v] = data['weight']
        adj_matrix[v][u] = data['weight']  # Ensure symmetry    

    # Solve TSP using brute force
    route, distance = graph.brute_force_tsp(adj_matrix)
    print("Shortest Route:", route)
    print("Total Distance:", distance)
    max_edge_weight = max(data['weight'] for _, _, data in graph.G.edges(data=True))
    print("Max Weight of Graph:", max_edge_weight)      
    
    # Plot the TSP route on the graph
    
    graph.plot_tsp_route(route)

    # Configure and solve the QUBO problem
    
    # With LocicalX Mixer - This is not working yet
    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, mixer='LogicalX', initial_state='OHE')

    # With mixer X - Standard formulation
    qubo_problem = QAOA_TSP_Maxcut(graph.G, config, TSP=True)

    # Print the number of qubits necessary to solve the problem
    qubo_problem.print_number_of_qubits()

    p = 1  # QAOA circuit depth

    optimal_params = qubo_problem.solve_problem(p)
    samples = qubo_problem.qubo_sample(optimal_params)

    # Visualize the solution

    valid_solution = sample_and_plot_histogram(
        samples,
        adj_matrix=adj_matrix,
        N=N,
        interpret_solution_fn=interpret_solution,
        top_n=30,
        var_names=qubo_problem.var_names
    )    

    # Draw the most sampled
    if valid_solution != []:
        draw_tsp_solution(graph, valid_solution)


if __name__ == "__main__":
    main()