import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the data files
merged_word_counts_file = "./merged_word_counts.tsv"
monthly_counts_file = "./monthly_counts.tsv"

merged_df = pd.read_csv(merged_word_counts_file, sep="\t")
monthly_counts_df = pd.read_csv(monthly_counts_file, sep="\t")

# Ensure Year_Month is consistent in both dataframes
merged_df['Year_Month'] = pd.to_datetime(merged_df['Year_Month'], format="%Y_%m")
monthly_counts_df['Year_Month'] = pd.to_datetime(monthly_counts_df['Year_Month'], format="%Y_%m")

# Merge the dataframes on Year_Month
combined_df = pd.merge(merged_df, monthly_counts_df, on='Year_Month', how='inner')

type='whole'    # whole topic_keyword

# Normalize word counts by whole_count for each media
normalized_columns = {}
for spec_topic, total in [('News7', f'{type}_news'), ('Papers10', f'{type}_papers'), ('Patents13', f'{type}_patents')]:

    print(spec_topic, total)

    normalized_col = f"{spec_topic}_Normalized"
    combined_df[normalized_col] = combined_df[spec_topic] / (combined_df[total] + 1e-6) # 특허에서 분모가 0인 경우가 존재하여 매우 작은값 더함
    normalized_columns[spec_topic] = normalized_col

print(combined_df)
for column in combined_df.columns:
    if combined_df[column].isna().any():
        print(f"Missing values in column: {column}")
        print(combined_df[combined_df[column].isna()])

missing_summary = combined_df.isna().sum()
print(missing_summary[missing_summary > 0])

# Perform time-series decomposition for normalized data
decompositions = {}
for media, column in normalized_columns.items():
    print(media, column)
    decomposition = seasonal_decompose(combined_df[column], model='additive', period=12)
    decompositions[media] = decomposition

    # Plot the decomposition results
    decomposition.plot()
    plt.suptitle(f"{media} Normalized Time-Series Decomposition", fontsize=16)
    plt.show()

# Plot the trends from the decompositions in a single plot
plt.figure(figsize=(12, 8))

for media, decomposition in decompositions.items():
    trend = decomposition.trend.dropna()  # Remove NaN values for plotting
    plt.plot(trend.index, trend, label=f"{media} Trend")

# plt.title("Normalized Trends Across Media", fontsize=16)
plt.xlabel("Year_Month", fontsize=14)
plt.ylabel(f"Normalized Count (Trend) {type}", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(f'norm_trend_comparison_{type}.png', dpi=300)
plt.show()
