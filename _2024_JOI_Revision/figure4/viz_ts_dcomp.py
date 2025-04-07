import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

# Load the merged data
merged_file = "./merged_word_counts.tsv"
merged_df = pd.read_csv(merged_file, sep="\t")

# Ensure Year_Month is sorted and set as the index for time-series analysis
merged_df = merged_df.sort_values("Year_Month")
merged_df['Year_Month'] = pd.to_datetime(merged_df['Year_Month'], format="%Y_%m")
merged_df.set_index('Year_Month', inplace=True)

# Drop rows with missing values for cleaner analysis
merged_df = merged_df.dropna()

# Perform time-series decomposition for each source
decompositions = {}
for col in ['News7', 'Papers10', 'Patents13']:
    decompositions[col] = seasonal_decompose(merged_df[col], model='additive', period=12)

# Plot decompositions for visualization
for col, decomposition in decompositions.items():
    decomposition.plot()
    plt.suptitle(f"{col} Time-Series Decomposition", fontsize=16)
    plt.show()


# Combine trend components of News, Papers, and Patents into a single DataFrame
trend_data = pd.DataFrame({
    "Year_Month": merged_df.index,
    "News_Trend": decompositions['News'].trend,
    "Papers_Trend": decompositions['Papers'].trend,
    "Patents_Trend": decompositions['Patents'].trend
}).dropna()  # Drop NaN values caused by decomposition

# Plot the trends with a single y-axis using log scale
plt.figure(figsize=(12, 8))

# Plot trends for News, Papers, and Patents on a single y-axis
plt.plot(trend_data['Year_Month'], trend_data['News_Trend'], label="News (Topic 7)", color="orange", marker='o', linewidth=2)
plt.plot(trend_data['Year_Month'], trend_data['Papers_Trend'], label="Papers (Topic 10)", color="red", marker='o', linewidth=2)
plt.plot(trend_data['Year_Month'], trend_data['Patents_Trend'], label="Patents (Topic 13)", color="blue", marker='o', linewidth=2)

# Add labels, legend, and title
plt.xlabel("Year", fontsize=20)
plt.ylabel("ToTal Word Count", fontsize=20)
# plt.title("Trend Comparison Across Different Document Types", fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("trend_comparison.png", dpi=300)
# Show the plot
plt.show()

