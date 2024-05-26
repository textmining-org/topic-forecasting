import pandas as pd

# Set the display precision for floating-point numbers
pd.options.display.float_format = '{:.10f}'.format

# Load the provided CSV file
model = 'a3tgcn2'
rnn_sub = 'lstm'
media = 'papers'  # patents, papers, news
file_path = f'./metrics_trng_{model}.csv'
data = pd.read_csv(file_path)

hidden_size = num_layers = K = embedd_dim = out_channels = None
if 'rnns' == model:
    if rnn_sub == 'lstm':
        if media == 'patents':
            hidden_size = 4
            num_layers = 3
        elif media == 'papers':
            hidden_size = 32
            num_layers = 1
        elif media == 'news':
            hidden_size = 4
            num_layers = 1
    elif rnn_sub == 'gru':
        if media == 'patents':
            hidden_size = 4
            num_layers = 2
        elif media == 'papers':
            hidden_size = 16
            num_layers = 4
        elif media == 'news':
            hidden_size = 4
            num_layers = 4
elif 'agcrn' == model:
    if media == 'patents':
        K = 2
        embedd_dim = 16
        out_channels = 8
    elif media == 'papers':
        K = 3
        embedd_dim = 4
        out_channels = 8
    elif media == 'news':
        K = 2
        embedd_dim = 4
        out_channels = 32
elif 'a3tgcn2' == model:
    if media == 'patents':
        out_channels = 16
    elif media == 'papers':
        out_channels = 8
    elif media == 'news':
        out_channels = 32

# Filter the data based on the given conditions: K=2, embedd_dim=16, out_channels=8, media='patents'
if 'rnns' == model:
    filtered_data = data[(data['hidden_size'] == hidden_size) & (data['num_layers'] == num_layers) & (
            data['media'] == media)]
elif 'agcrn' == model:
    filtered_data = data[
        (data['K'] == K) & (data['embedd_dim'] == embedd_dim) & (data['out_channels'] == out_channels) & (
                data['media'] == media)]
elif 'a3tgcn2' == model:
    filtered_data = data[(data['out_channels'] == out_channels) & (data['media'] == media)]

# Group the data by 'node_feature_type' and calculate the mean MSE and MAE for each group
grouped_data = filtered_data.groupby('node_feature_type')[['mse', 'mae']].mean().reset_index()
# Display the resulting grouped data
print(grouped_data)
