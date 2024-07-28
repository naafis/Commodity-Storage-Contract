import pandas as pd
import numpy as np
from math import log


# Load the dataset
def load_data(filepath):
    return pd.read_csv(filepath)


# Assign each FICO score to a bucket based on the boundaries
def assign_to_bucket(fico_score, boundaries):
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= fico_score < boundaries[i + 1]:
            return i
    return len(boundaries) - 2  # Assign to the last bucket if it doesn't fall in any


# Function to calculate log-likelihood
def log_likelihood(n, k):
    if n == 0:
        return 0
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * log(p) + (n - k) * log(1 - p)


# Dynamic Programming function to find the optimal bucket boundaries
def find_optimal_buckets(defaults, totals, r):
    max_fico = len(defaults)
    dp = [[[-10 ** 18, 0] for _ in range(max_fico)] for _ in range(r + 1)]

    for i in range(r + 1):
        for j in range(max_fico):
            if i == 0:
                dp[i][j][0] = 0
            else:
                for k in range(j):
                    if totals[j] == totals[k]:
                        continue
                    current_ll = log_likelihood(totals[j] - totals[k], defaults[j] - defaults[k])
                    if i == 1:
                        dp[i][j][0] = current_ll
                    else:
                        new_ll = dp[i - 1][k][0] + current_ll
                        if dp[i][j][0] < new_ll:
                            dp[i][j][0] = new_ll
                            dp[i][j][1] = k

    # Extract bucket boundaries
    boundaries = []
    k = max_fico - 1
    while r >= 0:
        boundaries.append(k + 300)
        k = dp[r][k][1]
        r -= 1

    boundaries.reverse()
    return dp[-1][-1][0], boundaries


def main():
    # File path for the dataset
    file_path = 'Task_3_and_4_Loan_Data.csv'

    # Load data
    df = load_data(file_path)

    # Extract FICO scores and default status
    fico_scores = df['fico_score'].to_list()
    defaults = df['default'].to_list()
    n = len(fico_scores)

    # Initialize counts for default and total records
    max_fico = 851
    default = [0] * max_fico
    total = [0] * max_fico

    # Count defaults and totals for each FICO score
    for i in range(n):
        score = int(fico_scores[i])
        default[score - 300] += defaults[i]
        total[score - 300] += 1

    # Compute cumulative defaults and totals
    for i in range(1, max_fico):
        default[i] += default[i - 1]
        total[i] += total[i - 1]

    # Define number of buckets
    num_buckets = 10

    # Find the optimal bucket boundaries
    max_log_likelihood, optimal_boundaries = find_optimal_buckets(default, total, num_buckets)

    # Output the results
    print(f"Maximum Log-Likelihood: {max_log_likelihood:.4f}")
    print("Optimal Bucket Boundaries:", optimal_boundaries)


if __name__ == "__main__":
    main()
