#treform 사용

import pandas as pd
import treform as ptm
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# paper / news / patent 중 선택
# paper
# eid	date	title	abstract	keywords    text
df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/data/papers_new_2022.csv', encoding="utf8").fillna("")
df.rename(columns={'cover_date': 'date'}, inplace=True)
df['keywords'] = df['keywords'].str.replace("|", "", )
df['text'] = df['title'] + " " + df['abstract'] + " " + df['keywords']
df['text'] = df['text'].str.lower()

#news
# df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/data/news_2022.csv', encoding="utf8").fillna("")
# df.rename(columns={'pubdate': 'date'}, inplace=True)
# df['text'] = df['Title'] + " " + df['full text']
# df['text'] = df['text'].str.lower()

#patent


#preprocessing
corpus = df['text']
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file='./stopwords/Stopword_Eng_Blockchain.txt'))
result = pipeline.processCorpus(corpus)

df['text'] = result # 이렇게 하면 문장단위로 list 나옴 ex. [[ , , , ], [ , , , ], [ , , , ], ....]

# document단위로 list통합하고 싶으면 아래 코드 추가 ex. [ , , , , ....]
# 아래 코드 추가안하면 sentense단위로 됨 
for i in range(len(df['text'])):
    df['text'][i] = sum(df['text'][i], [])
print(df['text'])

with open('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/pre_paper_3.pkl', "wb") as file:
    pickle.dump(df, file)


# 어떻게 생겼나 확인
df1 = pd.read_pickle('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/pre_paper_3.pkl')
#print(df1)
print(df1.head(40))


# # cooccurrence 구하기
# # 1. sentence 단위로 cooccurrence 구하기
# # 구조 변경(Sentence co-occurrence word를 찾기 위해 하나의 setence를 하나의 document로 변경)
# import re
# documents = []
# for doc in result:
#     for sent in doc:
#         sentence = ' '.join(sent)
#         sentence = re.sub('[^A-Za-z0-9가-힣_ ]+', '', sentence)
#         sentence = sentence.strip()
#         print(sentence)
#         if len(sentence) > 0:
#             documents.append(sentence)
#
# print(len(documents))
# co = ptm.cooccurrence.CooccurrenceWorker()
# co_result, vocab = co(documents)

# # 2. document단위로 cooccurrence 구하기
# import re
# documents = []
# for doc in result:
#     doc = re.sub('[^A-Za-z0-9가-힣_ ]+', '', str(doc))
#     doc = doc.strip()
#     print(doc)
#     if len(doc) > 0:
#         documents.append(doc)
#
# print(len(documents))
# co = ptm.cooccurrence.CooccurrenceWorker()
# co_result, vocab = co(documents)
#
# print(co_result)
# print(vocab)


# # co_result, word_hist, graph centrality 구하기
# graph_builder = ptm.graphml.GraphMLCreator()
# # mode is either with_threshold or without_threshod
# mode = 'with_threshold'
# if mode is 'without_threshold':
#     print(str(co_result))
#     print(str(vocab))
#     graph_builder.createGraphML(co_result, vocab, "test1.graphml")
# elif mode is 'with_threshold':
#     cv = CountVectorizer()
#     cv_fit = cv.fit_transform(documents)
#     word_list = cv.get_feature_names();
#     count_list = cv_fit.toarray().sum(axis=0)
#     word_hist = dict(zip(word_list, count_list))
#
#     print(str(co_result))
#     print(str(word_hist))
#
#     graph_builder.createGraphMLWithThreshold(co_result, word_hist, vocab, "test.graphml", threshold=35.0)
#     display_limit = 100
#     graph_builder.summarize_centrality(limit=display_limit)
#     title = '동시출현 기반 그래프'
#     file_name = 'test.png'
#     graph_builder.plot_graph(title, file=file_name)
