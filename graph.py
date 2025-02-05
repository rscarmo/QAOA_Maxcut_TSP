import networkx as nx
import random
import matplotlib.pyplot as plt
import itertools
import numpy as np
from plot_solutions import draw_maxcut_solution

class Graph:
    def __init__(self, num_nodes, weight_range=(1, 100), seed=None):
        """
        Initialize a random complete graph.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - weight_range: Range of edge weights (inclusive).
        - seed: Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        self.weight_range = weight_range
        self.seed = seed
        self.G = None
        self.pos = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._generate_random_complete_graph()

    def _generate_random_complete_graph(self):
        """Generate a random complete graph."""
        self.G = nx.complete_graph(self.num_nodes)
        for (u, v) in self.G.edges():
            self.G.edges[u, v]['weight'] = random.randint(self.weight_range[0], self.weight_range[1])

    def draw(self, with_labels=True, node_color='lightblue', edge_color='gray', 
             node_size=500, font_size=12):
        """Draw the graph."""
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)
        
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_color, width=1.5)

        if with_labels:
            nx.draw_networkx_labels(self.G, self.pos, font_size=font_size, font_color='black')

        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_color='red', font_size=10)

        plt.title("Random Complete Graph with Weighted Edges")
        plt.axis('off')
        plt.show()

    def calculate_total_distance(self, route, dist_matrix):
        total_distance = 0
        num_cities = len(route)
        for i in range(num_cities):
            from_city = route[i]
            to_city = route[(i + 1) % num_cities]
            total_distance += dist_matrix[from_city][to_city]
        return total_distance

    def brute_force_tsp(self, dist_matrix):
        num_cities = len(dist_matrix)
        cities = list(range(num_cities))
        starting_city = cities[0]
        other_cities = cities[1:]
        
        shortest_route = None
        minimal_distance = np.inf
        
        for perm in itertools.permutations(other_cities):
            current_route = [starting_city] + list(perm)
            current_distance = self.calculate_total_distance(current_route, dist_matrix)
            
            if current_distance < minimal_distance:
                minimal_distance = current_distance
                shortest_route = current_route
        
        return shortest_route, minimal_distance
    
    def plot_tsp_route(self, route, node_color='lightblue', edge_color='gray', 
                    node_size=500, font_size=12, route_color='green', route_width=2):
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)
        
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_color, width=1.5)
        nx.draw_networkx_labels(self.G, self.pos, font_size=font_size, font_color='black')
        
        # Draw the TSP route
        route_edges = []
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            route_edges.append((from_city, to_city))
        
        nx.draw_networkx_edges(self.G, self.pos, edgelist=route_edges, edge_color=route_color, width=route_width)
        
        plt.title("TSP Shortest Route")
        plt.axis('off')
        plt.show()    

    def brute_force_maxcut(self, G):
        """
        Brute force approach to find the maximum cut of a small graph.
        
        Returns:
            best_cost (float)
            best_partition (set): which nodes are in side A
            best_cut_edges (list): edges crossing the cut
        """
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        best_cost = float('-inf')
        best_partition = set()
        best_cut_edges = []
        
        for subset_int in range(1 << n):  # from 0 to 2^n - 1
            # Build a set of nodes for this subset
            current_partition = {
                nodes[i] for i in range(n) if (subset_int & (1 << i)) != 0
            }
            
            # Calculate cost
            cost = 0
            cut_edges = []
            for u, v, w in G.edges(data='weight', default=1):
                # If it crosses the partition
                if (u in current_partition) != (v in current_partition):
                    cost += w
                    cut_edges.append((u, v))
            
            if cost > best_cost:
                best_cost = cost
                best_partition = current_partition
                best_cut_edges = cut_edges[:]
        
        return best_cost, best_partition, best_cut_edges

    def draw_brute_force_maxcut_solution(self, G):
        """
        Find the maximum cut by brute force and draw it.
        """
        best_cost, best_partition, best_cut_edges = self.brute_force_maxcut(G)
        print(f"Brute force found best cut cost = {best_cost}")
        draw_maxcut_solution(G, best_partition, best_cut_edges, 
                            title="Best Cut by Brute Force")



# Example Usage
# if __name__ == "__main__":
#     N = 4  # Number of cities
#     weight_range = (10, 100)  # Weight range for edges
#     seed = 5 # Seed for reproducibility
    
#     # Generate and draw the random complete graph
#     G, num_edges = generate_random_complete_graph(N, weight_range, seed)
#     pos = draw_complete_graph(G)
    
#     # Create distance matrix from graph
#     adj_matrix = np.zeros((N, N), dtype=int)
#     for (u, v, data) in G.edges(data=True):
#         adj_matrix[u][v] = data['weight']
#         adj_matrix[v][u] = data['weight']  # Ensure symmetry
    
#     # Solve TSP using brute force
#     route, distance = brute_force_tsp(adj_matrix)
#     print("Shortest Route:", route)
#     print("Total Distance:", distance)
#     max_edge_weight = max(data['weight'] for _, _, data in G.edges(data=True))
#     print("Max Weight of Graph:", max_edge_weight)      
    
#     # Plot the TSP route on the graph
#     plot_tsp_route(G, route, pos)