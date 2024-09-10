import pandas as pd
from sklearn.model_selection import train_test_split
from bayesian_network import create_expert_graph, tree_hill_climbing
from mutual_information import calculate_mutual_information_matrix, plot_mutual_information_heatmap, visualize_network_with_weights

# Load dataset
data = pd.read_csv('heart.csv')

# Identify continuous variables and bin them
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for var in continuous_vars:
    data[var + '_binned'] = pd.qcut(data[var], q=3, labels=False, duplicates='drop')

data = data.drop(columns=continuous_vars)
train_data, validation_data = train_test_split(data, test_size=0.15, random_state=42)

# Initialize the expert graph
expert_graph = create_expert_graph(data.columns)

# Learn the structure using the expert graph as a starting point
bn_structure = tree_hill_climbing(train_data, expert_graph)

# Calculate mutual information
mi_matrix = calculate_mutual_information_matrix(train_data)

# Plot mutual information heatmap
plot_mutual_information_heatmap(mi_matrix)

# Visualize the Bayesian Network with weights
visualize_network_with_weights(bn_structure, mi_matrix)
