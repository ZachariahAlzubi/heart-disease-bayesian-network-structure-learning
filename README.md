# Bayesian Network Structure Learning

## Overview
This project demonstrates the learning of a Bayesian Network structure using an expert graph as a starting point and hill climbing with a BIC score. Additionally, it calculates and visualizes mutual information between variables.

## Files
- `bayesian_network.py`: Contains functions for BIC calculation and Bayesian network structure learning.
- `mutual_information.py`: Contains functions for calculating mutual information and visualizing the network.
- `main.py`: The main script that ties everything together and executes the learning process.

## Requirements
- NumPy
- pandas
- scikit-learn
- NetworkX
- seaborn
- matplotlib

## How to Run
1. Install the dependencies:
pip install numpy pandas scikit-learn networkx seaborn matplotlib


2. Run the main script:
python main.py


3. Make sure that the `heart.csv` dataset is available in the same directory as the script.
