import networkx as nx
import random
from tqdm.auto import tqdm


def generate_random_directed_graph(num_nodes: int, avg_degree: float = 2.0, p_dir: float = 0.5, seed: int = None):
    """
    Generate a random directed graph using an Erdős-Rényi G(n, p) model for the
    undirected backbone, then assign direction to edges with probability p_dir.
    Also ensures each node is not isolated (has at least one incoming or outgoing edge).
    """
    if seed is not None:
        random.seed(seed)
    
    # Probability p for the Erdős-Rényi model (undirected)
    if num_nodes > 1:
        p = avg_degree / (num_nodes - 1)
    else:
        p = 0.0
    
    undirected_graph = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=seed)
    
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(undirected_graph.nodes())
    
    # Assign directions to edges
    for (u, v) in undirected_graph.edges():
        if random.random() < p_dir:
            directed_graph.add_edge(u, v)
        else:
            directed_graph.add_edge(v, u)
    
    # Ensure no isolated nodes
    ensure_no_isolated_nodes(directed_graph)
    
    return directed_graph


def ensure_no_isolated_nodes(dg: nx.DiGraph):
    """
    For each node, if it has out_degree == 0, add a random outgoing edge.
    If it has in_degree == 0, add a random incoming edge.
    This guarantees that every node has at least one edge.
    """
    nodes_list = list(dg.nodes())
    for n in nodes_list:
        # Ensure at least one outgoing edge
        if dg.out_degree(n) == 0 and len(nodes_list) > 1:
            m = random.choice([x for x in nodes_list if x != n])
            dg.add_edge(n, m)
        # Ensure at least one incoming edge
        if dg.in_degree(n) == 0 and len(nodes_list) > 1:
            m = random.choice([x for x in nodes_list if x != n])
            dg.add_edge(m, n)


def assign_input_and_output_nodes(graph: nx.DiGraph, n_input: int, n_output: int, seed: int = None):
    """
    Randomly choose 'n_input' nodes to be input nodes and 'n_output' to be output nodes.
    """
    if seed is not None:
        random.seed(seed)
    all_nodes = list(graph.nodes())
    random.shuffle(all_nodes)
    input_nodes = all_nodes[:n_input]
    output_nodes = all_nodes[n_input : n_input + n_output]
    return input_nodes, output_nodes


def simulate_signals_with_random_termination(graph: nx.DiGraph,
                                             input_nodes,
                                             output_nodes,
                                             p_stop=0.1):
    """
    Simulate signal propagation with probability 'p_stop' that a node
    does NOT forward its signal to successors. (1 - p_stop) chance it forwards.
    
    Returns:
      - signal_history: a list of sets, each set is the set of active nodes at that step.
      - total_steps: the total number of steps until termination.
    """
    signal_history = []
    current_signals = set(input_nodes)
    step_count = 0
    
    while True:
        signal_history.append(current_signals)
        step_count += 1
        
        next_signals = set()
        for node in current_signals:
            # If it's an output node, signal terminates here
            if node in output_nodes:
                continue
            # Otherwise, forward with probability (1 - p_stop)
            for neighbor in graph.successors(node):
                if random.random() > p_stop:
                    next_signals.add(neighbor)
        
        if len(signal_history) % 1000 == 0:
            p_stop += 0.01
        
        if not next_signals:
            # Add the empty state and break
            signal_history.append(set())
            break
        
        current_signals = next_signals
    
    return signal_history, step_count


def draw_graph_state(graph,
                     pos,
                     active_nodes,
                     input_nodes,
                     output_nodes,
                     ax,
                     step_num):
    ax.clear()
    # Draw base nodes (light gray) and edges
    nx.draw_networkx_nodes(graph, pos, node_color="lightgray", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, ax=ax)
    
    # Highlight input nodes in green, output nodes in blue, active in red
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=input_nodes, node_color="green", ax=ax)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=output_nodes, node_color="blue", ax=ax)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=list(active_nodes), node_color="red", ax=ax)
    
    ax.set_title(f"Step {step_num}")
    ax.axis("off")


def run_experiments(node_sizes=[50, 100],
                    avg_degrees=[1.5, 2.0],
                    num_experiments=3,
                    n_input=2,
                    p_stop=0.1):
    """
    Run multiple experiments across given node_sizes and avg_degrees.
    For each combination, run 'num_experiments' trials.
    
    Returns:
      - results: a list of dicts, each with:
          {
            'node_size': ...
            'avg_degree': ...
            'experiment_id': ...
            'graph': ...
            'input_nodes': ...
            'output_nodes': ...
            'signal_history': ...
            'total_steps': ...
          }
      We store the entire graph & signal_history so we can replay the longest run.
    """
    results = []
    exp_count = 0
    
    for n in tqdm(node_sizes):
        for d in avg_degrees:
            for e in range(num_experiments):
                exp_count += 1
                # We vary seed for each run so we get different graphs
                
                # 1) Build graph
                G = generate_random_directed_graph(n, d, p_dir=0.5)
                
                n_output = n // 10
                # 2) Assign input & output
                input_nodes, output_nodes = assign_input_and_output_nodes(G,
                                                                          n_input,
                                                                          n_output)
                
                # 3) Simulate
                signal_history, total_steps = simulate_signals_with_random_termination(
                    G,
                    input_nodes,
                    output_nodes,
                    p_stop=p_stop
                )
                
                # Store results
                results.append({
                    'node_size': n,
                    'avg_degree': d,
                    'experiment_id': e,
                    'graph': G,
                    'input_nodes': input_nodes,
                    'output_nodes': output_nodes,
                    'signal_history': signal_history,
                    'total_steps': total_steps
                })
    
    return results