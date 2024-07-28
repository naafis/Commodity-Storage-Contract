import pandas as pd
import numpy as np

# Load the dataset
file_path = "Task_3_and_4_Loan_Data.csv"
data = pd.read_csv(file_path)

# Define bucket boundaries for the two ranges: 0-600 and 600-850
bucket_boundaries_1 = np.linspace(0, 600, 6)
bucket_boundaries_2 = np.linspace(600, 850, 6)


# Function to assign each FICO score to a bucket based on the new boundaries
def assign_to_new_bucket(fico_score):
    if fico_score <= 600:
        return assign_to_bucket(fico_score, bucket_boundaries_1)
    else:
        return assign_to_bucket(fico_score, bucket_boundaries_2) + len(bucket_boundaries_1) - 1


# Helper function to assign bucket based on boundaries
def assign_to_bucket(fico_score, boundaries):
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= fico_score < boundaries[i + 1]:
            return i
    return len(boundaries) - 2  # Assign to the last bucket if it doesn't fall in any


# Assign each record to a new bucket
data['new_bucket'] = data['fico_score'].apply(assign_to_new_bucket)

# Calculate the number of records and defaults in each new bucket
new_bucket_stats = data.groupby('new_bucket').agg(
    total_records=('default', 'size'),
    total_defaults=('default', 'sum')
)

# Calculate the probability of default for each new bucket
new_bucket_stats['probability_of_default'] = new_bucket_stats['total_defaults'] / new_bucket_stats['total_records']

# Calculate the log-likelihood for the new bucket setup
new_bucket_stats['log_likelihood'] = new_bucket_stats.apply(
    lambda row: row['total_defaults'] * np.log(row['probability_of_default']) +
                (row['total_records'] - row['total_defaults']) * np.log(1 - row['probability_of_default']),
    axis=1
)

# Calculate the total log-likelihood
new_total_log_likelihood = new_bucket_stats['log_likelihood'].sum()

# Output the results
print(new_bucket_stats)
print(f"Total Log-Likelihood: {new_total_log_likelihood}")
