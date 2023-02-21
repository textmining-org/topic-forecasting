#treform 사용

import pandas as pd
import treform as ptm
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

#paper / news / patent 중 선택

# paper : eid	date	title	abstract	keywords    text
# df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/data/papers_new_2022.csv', encoding="utf8").fillna("")
# df.rename(columns={'cover_date': 'date'}, inplace=True)
# df['keywords'] = df['keywords'].str.replace("|", "", )
# df['text'] = df['title'] + " " + df['abstract'] + " " + df['keywords']
# df['text'] = df['text'].str.lower()

# news : pubdate / Title / full text
# df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/data/news_new.csv', encoding="utf8").fillna("")
# df.rename(columns={'pubdate': 'date'}, inplace=True)
# df['text'] = df['Title'] + " " + df['full text']
# df['text'] = df['text'].str.lower()

# patent : date_original	date	title	abstract
df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/data/patent_new.csv', encoding='cp949').fillna("")
#df.rename(columns={'pubdate': 'date'}, inplace=True)
df['text'] = df['title'] + " " + df['abstract']
df['text'] = df['text'].str.lower()


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
for i in range(len(df['text'])):
    df['text'][i] = sum(df['text'][i], [])
print(df['text'])

with open('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/pre_patents_sen.pkl', "wb") as file:
    pickle.dump(df, file)


# 어떻게 생겼나 확인
df1 = pd.read_pickle('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/pre_patents_sen.pkl')
#print(df1)
print(df1.head(40))

