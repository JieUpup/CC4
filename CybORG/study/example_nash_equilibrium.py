import nashpy as nash
import numpy as np

# Define the payoff matrices for two players
# Player 1's payoff matrix (rows)
A = np.array([[3, 0], [5, 1]])

# Player 2's payoff matrix (columns)
B = np.array([[3, 5], [0, 1]])

# Create the bi-matrix game
game = nash.Game(A, B)

# Compute Nash equilibria using the support enumeration algorithm
equilibria = list(game.support_enumeration())

# Print all Nash equilibria
print("Nash Equilibria:")
for eq in equilibria:
    print(f"Player 1: {eq[0]}, Player 2: {eq[1]}")

