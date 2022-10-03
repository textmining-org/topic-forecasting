import os

import pandas as pd
import plotly.graph_objects as go
import tomotopy as tp
from plotly.subplots import make_subplots

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


def calc_coehrence_perplexity(target_name, method_name, num_topics_range):
    num_topics_list = list(num_topics_range)
    # perplexity/coherence 계산
    coherence_metric = 'c_v'  # u_mass(0에 가까울수록 일관성 높음), c_uci, c_npmi, c_v(0과1사이, 0.55정도 수준)
    perplexities = []
    coherences = []
    for num_topics in num_topics_list:
        model_save_path = os.path.join('./opt_num_topics', target_name + '_' + method_name + '_' + num_topics + '.bin')
        model = tp.LDAModel.load(model_save_path)
        coherence = tp.coherence.Coherence(model, coherence=coherence_metric)
        print(
            'num topics: {}\tperplexity: {}\tcoherence: {}'.format(num_topics, model.perplexity, coherence.get_score()))
        perplexities.append(model.perplexity)
        coherences.append(coherence.get_score())

    # csv 저장
    columns = ['topic number', 'perplexity', 'coherence', 'metric']
    df_result = pd.DataFrame(columns=columns)
    df_result = df_result.append(
        pd.DataFrame((zip(num_topics_list, perplexities, coherences, [coherence_metric] * len(num_topics_list))),
                     columns=columns), ignore_index=True)
    print(df_result)

    result_file = f'./results/{target_name}_{method_name}_perplexity_coherence.csv'
    df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

    # plot perplexity/coherence
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
    datasets_path = os.path.abspath('../_datasets')
    print(datasets_path)

    num_topics_range = range(2, 101, 2)

    make_model(datasets_path, 'patents', 'lda', num_topics_range)
    make_model(datasets_path, 'papers', 'lda', num_topics_range)
    make_model(datasets_path, 'news', 'lda', num_topics_range)
