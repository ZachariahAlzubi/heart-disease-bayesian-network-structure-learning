import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mutual_info_score

def calculate_bic(data, graph):
    score = 0
    n = len(data)
    
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        if parents:
            if len(parents) == 1:
                parent_series = data[parents[0]]
            else:
                parent_series = data[parents].astype(str).agg('-'.join, axis=1)

            contingency_table = pd.crosstab(parent_series, data[node], margins=True)
            log_likelihood = 0
            for parent_state, row in contingency_table.iterrows():
                if parent_state == 'All':
                    continue
                row_sum = row['All']
                for state, count in row.items():
                    if state != 'All' and row_sum > 0:
                        probability = count / row_sum
                        if probability > 0:
                            log_likelihood += count * np.log(probability)

        else:
            counts = data[node].value_counts()
            total = counts.sum()
            log_likelihood = sum(count * np.log(count / total) for count in counts if count > 0)
        
        num_params = (data[node].nunique() - 1) * np.prod([data[parent].nunique() for parent in parents])
        score += log_likelihood
        score -= num_params / 2 * np.log(n)

    return score

def create_expert_graph(columns):
    expert_graph = nx.DiGraph()
    expert_edges = [
        ("age_binned", "trestbps_binned"), ("age_binned", "restecg"), ("age_binned", "fbs"), ("age_binned", "chol_binned"),
        ("sex", "chol_binned"),
        ("chol_binned", "ca"),
        ("ca", "slope"),
        ("slope", "oldpeak_binned"),
        ("oldpeak_binned", "exang"), ("oldpeak_binned", "target"),
        ("exang", "target"),
        ("fbs", "target"),
        ("restecg", "slope"),
        ("thal", "exang"), ("thal", "thalach_binned"), ("thal", "restecg"),
        ("thalach_binned", "exang"),
        ("trestbps_binned", "restecg"), ("trestbps_binned", "cp"),
        ("cp", "target")
    ]
    expert_graph.add_edges_from(expert_edges)
    return expert_graph

def tree_hill_climbing(data, initial_graph):
    best_graph = initial_graph
    best_score = float('-inf')
    target_node = data.columns[-1]
    
    improved = True
    while improved:
        improved = False
        for node in set(data.columns) - {target_node}:
            potential_parents = set(data.columns) - {node} - set(best_graph.predecessors(node))
            for parent in potential_parents:
                new_graph = best_graph.copy()
                new_graph.add_edge(parent, node)
                if nx.is_directed_acyclic_graph(new_graph):
                    new_score = calculate_bic(data, new_graph)
                    if new_score > best_score:
                        best_graph = new_graph
                        best_score = new_score
                        improved = True
                        break
    return best_graph
