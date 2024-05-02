import os
import pandas as pd
import plotly.graph_objects as go
import tomotopy as tp
from plotly.subplots import make_subplots
import argparse

from commons import exec_lda_modeling, exec_dmr_modeling

def make_model(datasets_path, target_name, method_name, num_topics_range):
    preprocessed_path = os.path.join(datasets_path, target_name + '.pkl')
    df_datasets = pd.read_pickle(preprocessed_path)
    timestamps = df_datasets['date'].astype(str).values.tolist()
    documents = df_datasets['text'].values.tolist()

    num_topics_list = list(num_topics_range)
    for num_topics in num_topics_list:
        if method_name == 'lda':
            model = exec_lda_modeling(documents, num_topics)
        elif method_name == 'dmr':
            model = exec_dmr_modeling(documents, timestamps, num_topics)

        print(f'##### number of topics: {num_topics}')
        model.save(f'./opt_num_topics/{target_name}_{method_name}_{num_topics}.bin', full=True)

def calc_coherence_perplexity(target_name, method_name, num_topics_range):
    num_topics_list = list(num_topics_range)
    coherence_metric = 'c_v'
    data_list = []
    for num_topics in num_topics_list:
        model_save_path = os.path.join('./opt_num_topics', f'{target_name}_{method_name}_{num_topics}.bin')
        if method_name == 'lda':
            model = tp.LDAModel.load(model_save_path)
        elif method_name == 'dmr':
            model = tp.DMRModel.load(model_save_path)
        coherence = tp.coherence.Coherence(model, coherence=coherence_metric)
        print(f'num topics: {num_topics}\tperplexity: {model.perplexity}\tcoherence: {coherence.get_score()}')
        data_list.append({
            'topic number': num_topics,
            'perplexity': model.perplexity,
            'coherence': coherence.get_score(),
            'metric': coherence_metric
        })

    # Convert the list of dictionaries to a DataFrame
    df_result = pd.DataFrame(data_list)

    result_file = f'./opt_num_topics/{target_name}_{method_name}_perplexity_coherence.csv'
    df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

    # Plot perplexity/coherence
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    trace1 = go.Scatter(x=df_result['topic number'], y=df_result['perplexity'], name='Perplexity')
    trace2 = go.Scatter(x=df_result['topic number'], y=df_result['coherence'], name='Coherence')
    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace2, secondary_y=True)
    fig.update_layout(title_text=f'{target_name} Perplexity and Coherence')
    fig.update_yaxes(title_text='Perplexity', secondary_y=False)
    fig.update_yaxes(title_text='Coherence', secondary_y=True)
    fig.update_xaxes(title_text='Number of topics')
    fig.write_html(os.path.splitext(result_file)[0] + '.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', default='papers')
    parser.add_argument('--method_name', default='dmr')
    args = parser.parse_args()

    datasets_path = os.path.abspath('../_datasets')
    print(datasets_path)

    num_topics_range = range(2, 51, 2)

    make_model(datasets_path, args.target_name, args.method_name, num_topics_range)
    calc_coherence_perplexity(args.target_name, args.method_name, num_topics_range)
