import tomotopy as tp
import sys
import pandas as pd
import numpy as np

# LDA 모델 생성 및 학습
def exec_lda_modeling(documents, topic_number, min_cf=3, rm_top=5, iter=1500):
    model = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top, k=topic_number)
    for document in documents:
        model.add_doc(document)
    model.burn_in = 100

    model.train(0)
    print('Num docs:', len(model.docs), ', Vocab size:', model.num_vocabs, ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, iter, 10):
        model.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

    print('Perplexity: {}'.format(model.perplexity))

    return model

def exec_dmr_modeling(documents, timestamps, topic_number, min_cf=3, rm_top=5, iter=1500):
    model = tp.DMRModel(tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top, k=topic_number)
    for timestamp, document in zip(timestamps, documents):
        model.add_doc(document, metadata=timestamp)
    model.burn_in = 100

    model.train(0)
    print('Num docs:', len(model.docs), ', Vocab size:', model.num_vocabs, ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, iter, 10):
        model.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

    # Perplexity(혼란도) : 언어 모델의 비교를 위한 정략적 측정 방법, 단일 단어를 예측하는 것에 대한 불확실성
    #                     그 값이 최소화되는 topic 개수의 경우가 가장 적절한 topic 수
    print('Perplexity: {}'.format(model.perplexity))

    return model

# 연도별 topic score 계산 및 저장
def dmr_topic_scoring(model):

    df_topic_score = pd.DataFrame()
    # k : the number of topics
    for topic_number in range(model.k):
        print('Topic #{}'.format(topic_number))

        lambdas = model.lambdas[topic_number]
        median_val = np.median(lambdas)
        max_val = np.max(lambdas)
        min_val = np.min(lambdas)

        features = []
        # f : the number of metadata features
        for metadata_number in range(model.f):
            orginal_val = model.lambdas[topic_number][metadata_number]
            new_val = abs(max_val) + orginal_val + abs(median_val)
            features.append(new_val)

        # topic별 추가
        df_topic_score = df_topic_score.append(pd.Series(features), ignore_index=True)

        print("median " + str(median_val) + " : " + str(max_val) + " : " + str(min_val))
        for word, prob in model.get_topic_words(topic_number):
            print('\t', word, prob, sep='\t')

    df_topic_score.columns = model.metadata_dict

    return df_topic_score

# topic labeler 생성
def get_topic_labeler(model):
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(model)
    labeler = tp.label.FoRelevance(model, cands, min_df=5, smoothing=1e-2, mu=0.25)
    return labeler
