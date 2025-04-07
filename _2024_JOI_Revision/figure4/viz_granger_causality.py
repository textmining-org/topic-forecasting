import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import csv

# Load the merged data
merged_file = "./merged_word_counts.tsv"
merged_df = pd.read_csv(merged_file, sep="\t")

# Ensure Year_Month is sorted and set as the index for time-series analysis
merged_df = merged_df.sort_values("Year_Month")
merged_df['Year_Month'] = pd.to_datetime(merged_df['Year_Month'], format="%Y_%m")
merged_df.set_index('Year_Month', inplace=True)

# Drop rows with missing values for cleaner analysis
merged_df = merged_df.dropna()

# Granger causality test
# Initialize a dictionary to store Granger causality results
granger_results = {}

# Define relationships to test
relationships = [
    ('News7', 'Papers10'),     # News -> Papers
    ('News7', 'Patents13'),    # News -> Patents
    ('Papers10', 'Patents13'),  # Papers -> Patents
    ('Papers10', 'News7'),  # Patents -> Papers
    ('Patents13', 'News7'),  # Patents -> Papers
    ('Patents13', 'Papers10')  # Patents -> Papers

]

# Perform Granger Causality tests for each relationship
for source, target in relationships:
    print(f"Testing Granger Causality: {source} → {target}")
    # Run Granger causality test
    result = grangercausalitytests(merged_df[[source, target]], maxlag=12, verbose=True)
    granger_results[f"{source}_to_{target}"] = result

    # Extract p-values for each lag
    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, 13)]

    # Plot p-values
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, 13), p_values, marker='o', label=f"{source} → {target}")
    plt.axhline(y=0.05, color='r', linestyle='--', label="Significance Level (0.05)")
    # plt.title(f"Granger Causality Test: {source} -> {target}")
    plt.xlabel("Lag", fontsize=14)
    plt.ylabel("p-value", fontsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'granger_causality_test_{source}_{target}.png', dpi=300)
    plt.show()


# Save results to CSV
csv_file = "granger_causality_p_values.csv"
header = ['Source', 'Target'] + [f"Lag_{lag}" for lag in range(1, 13)]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for source, target in relationships:
        p_values = [granger_results[f"{source}_to_{target}"][lag][0]['ssr_ftest'][1] for lag in range(1, 13)]
        row = [source, target] + p_values
        writer.writerow(row)

print(f"Granger causality p-values saved to {csv_file}")


