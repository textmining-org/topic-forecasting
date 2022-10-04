from commons import exec_lda_modeling, exec_dmr_modeling, get_topic_labeler, dmr_topic_scoring
import os
import pandas as pd


def exec_topic_modeling(datasets_path, target_name, method_name, num_topics):
    preprocessed_path = os.path.join(datasets_path, target_name + '.pkl')
    model_save_path = os.path.join('./models', target_name + '_' + method_name + '.bin')
    topic_keywords_save_path = os.path.join('./results', target_name + '_' + method_name + '_topic_keywords.csv')
    topic_score_save_path = os.path.join('./results', target_name + '_' + method_name + '_topic_score.csv')

    df_datasets = pd.read_pickle(preprocessed_path)
    timestamps = df_datasets['date'].astype(str).values.tolist()
    documents = df_datasets['text'].values.tolist()

    if method_name == 'lda':
        model = exec_lda_modeling(documents, num_topics)
    elif method_name == 'dmr':
        model = exec_dmr_modeling(documents, timestamps, num_topics)
    model.save(model_save_path, True)

    # topic label, keyword
    labeler = get_topic_labeler(model)
    df_topic_keywords = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
    for index, topic_number in enumerate(range(model.k)):
        label = ' '.join(label for label, score in labeler.get_topic_labels(topic_number, top_n=5))
        keywords = ' '.join(keyword for keyword, prob in model.get_topic_words(topic_number))
        df_topic_keywords.loc[index] = [topic_number, label, keywords]
    df_topic_keywords.to_csv(topic_keywords_save_path, encoding='utf-8-sig')

    if method_name == 'dmr':
        # timestamp별 topic score 계산 및 저장
        df_topic_score = dmr_topic_scoring(model)
        print(df_topic_score)
        df_topic_score.to_csv(topic_score_save_path, encoding='utf-8-sig')


if __name__ == '__main__':
    datasets_path = os.path.abspath('../_datasets')
    print(datasets_path)

    # LDA num_topics : patents=20, papers=8, news=
    # DMR num_topics : patents=14(12), papers=8(10), news=(18)
    exec_topic_modeling(datasets_path, 'papers', 'dmr', 8)
