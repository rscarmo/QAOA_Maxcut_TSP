import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import networkx as nx
import itertools
import numpy as np
import math
import pdb


def sample_to_dict(sample, var_names):
    """
    Convert a Qiskit 'SolutionSample' to a {var_name: bit_value} dict.
    """
    # sample.x is an array/list of 0/1 in the same order as 'var_names'
    # We cast each bit to int just for safety.
    try:
        return {
            name: int(val)
            for name, val in zip(var_names, sample.x)
        }
    except:
        return {
            name: int(val)
            for name, val in zip(var_names, sample)
        }        


def interpret_solution(solution_dict, N):
    """
    Interpret a TSP solution with city 0 fixed at position 0 and 
    x_{i}_{p} = 1 meaning city i is placed at position p (1..N-1).
    
    Returns:
        tour (list): [0, city_for_position_1, city_for_position_2, ..., city_for_position_(N-1)]
        or None if invalid.
    """
    # We fix city 0 at position 0
    tour = [0] + [None]*(N-1)  # e.g. for N=4, tour = [0, None, None, None]
    
    # Counters to ensure each city & each position is used exactly once
    city_counts = [0]*N            # city_counts[i] => how many times city i appears
    position_counts = [0]*N        # position_counts[p] => how many times position p is occupied
    
    for i in range(1, N):
        for p in range(1, N):  # now p goes from 1..N-1 inclusive
            var_name_x_ip = f"x_{i}_{p}"
            if solution_dict.get(var_name_x_ip, 0) == 1:
                # Assign city i to position p
                city_counts[i] += 1
                position_counts[p] += 1
                
                # If city or position is already used more than once => invalid
                if city_counts[i] > 1 or position_counts[p] > 1:
                    return None
                
                # Place city i in the tour array at index p
                tour[p] = i
    
    # Check that each city i=1..(N-1) is used exactly once
    if any(city_counts[i] != 1 for i in range(1, N)):
        return None
    
    # Check that each position p=1..(N-1) is used exactly once
    if any(position_counts[p] != 1 for p in range(1, N)):
        return None
    
    return tour




def sample_and_plot_histogram_tsp(samples, adj_matrix, N, interpret_solution_fn,
                              top_n=30, var_names=None):
    """
    Interpret QUBO samples, validate solutions, and plot a histogram of the most sampled valid bitstrings.

    Parameters:
    - samples: Dictionary de {bitstring: frequency} vindo do Qiskit (ou outro sampler).
    - adj_matrix: Matriz de adjacência do grafo.
    - N: Número de nós do grafo.
    - Delta: Restrição de grau máximo (se aplicável).
    - interpret_solution_fn: Função que, dado um dicionário de variáveis -> valores, 
      retorne a solução interpretada (por exemplo, um conjunto de arestas MST).
    - top_n: Número de soluções mais comuns para mostrar no histograma.
    - var_names: Lista/ordem de variáveis, caso seja preciso mapear bits do bitstring.
    - v0: (opcional) se precisar excluir ou tratar um vértice específico, etc.

    Returns:
    - most_common_valid_solutions: Lista das soluções mais comuns (até `top_n`),
      onde cada item é (edges_solution, freq, [lista de bitstrings]).
    """
    
    # -------------------------------------------------------------------------
    # 1) Agregador para as soluções válidas: soma de frequências e bitstrings
    # -------------------------------------------------------------------------
    aggregated_solutions = defaultdict(lambda: {"freq": 0, "bitstrings": []})
    
    for bitstring, frequency in samples.items():
        # 1.1) Convertemos o bitstring em dicionário var->valor
        solution_dict = sample_to_dict(bitstring, var_names)
        converted_dict = {var.name: val for var, val in solution_dict.items()}
       

        # 1.2) Interpretamos a solução (por ex, extrair arestas do MST)
        tsp_solution = interpret_solution_fn(converted_dict, N)
        
        # Se a função interpretou e validou de fato (pode conter None se inválida)
        # 'tsp_solution' aqui deve ser algo como uma lista de edges (u,v,w)
        if tsp_solution and all(e is not None for e in tsp_solution):
            edges_tuple = tuple(tsp_solution)
            
            # 1.3) Acumule na nossa estrutura
            # Frequência multiplicada se você quiser "ampliar" a escala.
            # Usando frequency*10000 como no seu exemplo:
            freq_scaled = frequency * 10000
            aggregated_solutions[edges_tuple]["freq"] += freq_scaled
            aggregated_solutions[edges_tuple]["bitstrings"].append(bitstring)

    if not aggregated_solutions:
        print("No valid TSP solutions were found.")
        return []

    # -------------------------------------------------------------------------
    # 2) Ordenar as soluções por frequência (decrescente) e pegar top_n
    # -------------------------------------------------------------------------
    # aggregated_solutions.items() = [(edges_tuple, {"freq": X, "bitstrings": [...]})]
    sorted_agg = sorted(
        aggregated_solutions.items(),
        key=lambda item: item[1]["freq"],
        reverse=True
    )
    # Reduzimos aos top_n
    sorted_agg = sorted_agg[:top_n]

    # Montamos a lista final no formato que você quer exibir/devolver:
    # (edges_solution, freq, bitstrings)
    most_common_valid_solutions = [
        (edges_tuple, data["freq"], data["bitstrings"])
        for edges_tuple, data in sorted_agg
    ]

    # -------------------------------------------------------------------------
    # 3) Plotar histograma
    # -------------------------------------------------------------------------
    labels = [f"Solution {i+1}" for i in range(len(most_common_valid_solutions))]
    frequencies = [item[1] for item in most_common_valid_solutions]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, frequencies, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Most Sampled Valid Solutions")
    plt.xlabel("Solutions")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()  # descomente se quiser exibir diretamente

    print("\nOs tours mais frequentes são:")
    for i, (tour, frequency, bitstring_list) in enumerate(most_common_valid_solutions, start=1):
        cost = sum(adj_matrix[tour[i], tour[(i + 1) % N]] for i in range(N))
        print(f"\nTour {i}: {tour}")
        print(f"Frequência: {frequency}")
        print(f"Custo total: {cost:.2f}")
        print("Bitstrings that produced this solution:")
        for bs in bitstring_list:
            print("  ", bs)
        print("-"*50)        

    # Interpretar o resultado
    most_common_tour, frequency, bitstring_list = most_common_valid_solutions[0]
    cost = sum(adj_matrix[most_common_tour[i], most_common_tour[(i + 1) % N]] for i in range(N))

    print("\nInterpretação do tour mais frequente:")
    print(f"O tour mais frequente encontrado pelo QAOA é {most_common_tour}, com um custo total de {cost:.2f}.")
    print("Este tour é a solução ótima encontrada para o problema do Caixeiro Viajante dado.")        

    return most_common_valid_solutions


def draw_tsp_solution(graph, solution_list, title="Traveling Salesman Solution"):
    """
    Visualize the TSP solution based on binary variable assignments such as x_{i}_{p} = 1.
    
    Parameters:
    - graph: An instance of your QAOA_TSP_Maxcut class, which has:
        - G: the underlying networkx Graph
        - n (or num_cities): number of cities
    - solution_dict: A dictionary { 'x_{i}_{p}': 0/1, ... } from Qiskit or from your aggregator.
    - title: Title of the plot.
    """
    #    Interpret the solution using the existing logic
    #    The interpret_solution function returns a list `tour = [0, cityA, cityB, ...]`
    #    or None if invalid. For example:
    #
    #    def interpret_solution(solution_dict, adj_matrix, N, Delta):
    #        ...
    #        return tour  # e.g. [0, 2, 3, 1], etc.
    #
    adj_matrix = nx.to_numpy_array(graph.G)  
    N = graph.num_nodes        
    tour = solution_list[0][0]    

    print(tour)
    
    # tour = interpret_solution(solution_dict, N)
    # if tour is None:
    #     print("Invalid TSP solution (it does not correspond to a valid route).")
    #     return
    
    # Build edges of the route
    route_edges = list(zip(tour, tour[1:]))
    # Let's close the loop:
    route_edges.append((tour[-1], tour[0]))
    
    # 2) Prepare a layout (or use a stored layout if you have one)
    pos = getattr(graph, 'pos', None)
    if pos is None:
        pos = nx.spring_layout(graph.G, seed=42)
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    
    # Draw all nodes
    nx.draw_networkx_nodes(graph.G, pos, node_color='lightblue', node_size=500)
    
    # Draw the TSP route edges in red
    nx.draw_networkx_edges(graph.G, pos, edgelist=route_edges, edge_color='red', width=2)
    
    # Draw the other edges (not used in the route) as dotted
    unused_edges = set(graph.G.edges()) - set(map(lambda e: tuple(sorted(e)), map(lambda x: tuple(sorted(x)), route_edges)))
    # Because (u,v) and (v,u) are the same in an undirected graph, handle them consistently:
    # We can unify them by always sorting each edge's tuple:
    G_edges_sorted = {tuple(sorted(e)) for e in graph.G.edges()}
    route_edges_sorted = {tuple(sorted(e)) for e in route_edges}
    dotted_edges = [tuple(e) for e in (G_edges_sorted - route_edges_sorted)]
    
    nx.draw_networkx_edges(graph.G, pos, edgelist=dotted_edges, style='dotted', edge_color='gray')
    
    # We'll label them as "city (pos p)"
    labels = {}
    for p, city in enumerate(tour):
        # e.g. "City 0 (pos 0)", "City 2 (pos 1)", ...
        labels[city] = f"{city} (pos {p})"
    
    nx.draw_networkx_labels(graph.G, pos, labels, font_size=12, font_color='black')
    
    plt.title(title)
    plt.axis("off")
    plt.show()


def interpret_maxcut_solution(bitstring, G):
    """
    Interpret a bitstring as a MaxCut partition.

    Parameters:
        bitstring (str): A string of '0's and '1's representing the partition (e.g., '10101').
        G (nx.Graph): The graph with .edges(data='weight').

    Returns:
        cost (float): The total cut value.
        cut_edges (list[tuple]): List of edges that cross the cut.
        partition (set): Set of nodes in the partition (corresponding to '1's in the bitstring).
    """
    # Convert the bitstring into a set of nodes in the partition (corresponding to '1')
    partition = {i for i, bit in enumerate(bitstring) if bit == '1'}

    cut_edges = []
    cost = 0.0

    # For each weighted edge, check if it crosses the partition
    for u, v, w in G.edges(data="weight", default=1):
        if (u in partition) != (v in partition):  # crosses the partition
            cost += w
            cut_edges.append((u, v))

    return cost, cut_edges, partition


def sample_and_plot_maxcut_histogram(samples, G, top_n=10):
    """
    Interpret MaxCut QAOA samples, compute cut values, and plot a histogram.

    Parameters:
        samples (dict): A dictionary {bitstring: frequency}.
        G (nx.Graph): The graph for MaxCut.
        top_n (int): Number of top solutions to display.

    Returns:
        most_common_solutions: List of (cost, frequency, bitstring, cut_edges).
    """
    # Store solutions with their computed costs and frequencies
    solution_data = []

    for bitstring, freq in samples.items():
        # Interpret each bitstring
        cost, cut_edges, partition = interpret_maxcut_solution(bitstring, G)
        solution_data.append((cost, freq, bitstring, cut_edges))

    # Sort by cost (descending) and frequency (descending)
    sorted_solutions = sorted(solution_data, key=lambda x: (-x[1]))

    # Select the top_n solutions
    top_solutions = sorted_solutions[:top_n]

    # Prepare data for the histogram
    labels = [f"{bitstring}" for _,_,bitstring,_ in top_solutions]
    frequencies = [data[1] for data in top_solutions]

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.bar(labels, frequencies, color="skyblue")
    plt.title(f"Top {top_n} Most Frequent MaxCut Solutions")
    plt.xlabel("Cut #")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Print details of the top solutions
    print("\nTop MaxCut Solutions:")
    for i, (cost, freq, bitstring, cut_edges) in enumerate(top_solutions, start=1):
        print(f"Solution {i}:")
        print(f"  Bitstring: {bitstring}")
        print(f"  Cost: {cost}")
        print(f"  Frequency: {freq}")
        print(f"  Cut Edges: {cut_edges}")
        print("-" * 50)

    return top_solutions


def draw_maxcut_solution(G, partition, cut_edges, title="MaxCut Solution"):
    """
    Draws the graph with two colors for the two partitions, 
    and highlights the cut edges.
    
    partition: set of nodes that are in side A (others are in side B)
    cut_edges: list of edges that cross the cut
    """
    pos = nx.spring_layout(G, seed=42)  # or any layout
    color_map = []
    for node in G.nodes():
        if node in partition:
            color_map.append('red')
        else:
            color_map.append('blue')
    
    plt.figure(figsize=(8,6))
    
    # Draw all edges first with a lighter style
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Highlight cut edges
    nx.draw_networkx_edges(
        G, pos, edgelist=cut_edges, edge_color='green', width=2
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=600)
    nx.draw_networkx_labels(G, pos, font_color='white')
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def pick_and_draw_a_top_solution(G, top_solutions):
    # Suppose you pick the #1 top solution
    (cost, freq, bs_list, cut_edges) = top_solutions[0]
    best_bitstring = bs_list  # pick the first bitstring that gave this solution

    # Directly derive the partition from the bitstring
    partition = {i for i, bit in enumerate(best_bitstring) if bit == '1'}
    
    print(f"Best Bitstring: {best_bitstring}")
    print(f"Partition: {partition}")
    print(f"Cut Edges: {cut_edges}")
    print(f"Cost: {cost}, Frequency: {freq}")

    # Draw the solution
    draw_maxcut_solution(G, partition, cut_edges, title="Top MaxCut QAOA Solution")

