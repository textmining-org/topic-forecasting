import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import string
import pathlib
import pickle
import re

import pandas as pd
import treform as ptm
import os


def data(datasets_path, target_name):
    data_path = os.path.join(datasets_path, target_name + '.csv')
    df = pd.read_csv(data_path, encoding='utf-8').fillna("")
    if target_name == 'patents_new': # patent : date_original	date	title	abstract
        df['text'] = df['title'] + " " + df['abstract']
        df['text'] = df['text'].str.lower()
        return df

    elif target_name == 'news_new': # news : pubdate / Title / full text
        df.rename(columns={'pubdate': 'date'}, inplace=True)
        df['text'] = df['Title'] + " " + df['full text']
        df['text'] = df['text'].str.lower()
        return df

    elif target_name == 'papers_new': # paper : eid	date	title	abstract	keywords    text
        df.rename(columns={'cover_date': 'date'}, inplace=True)
        df['keywords'] = df['keywords'].str.replace("|", "", )
        df['text'] = df['title'] + " " + df['abstract'] + " " + df['keywords']
        df['text'] = df['text'].str.lower()
        return df


def _preprocess_text(txt, stopword_file):
    txt = txt.lower()
    txt = re.sub('[^A-Za-z0-9가-힣_ ]+', '', txt)

    # tokenize
    words = word_tokenize(txt)

    # stopwords filtering
    stopwords_def = stopwords.words('english')  # nltk default stopwords
    stopwords_ctm = [word.strip() for word in open(stopword_file, encoding='utf-8')]
    stopwords_def.extend(stopwords_ctm)
    words_flt = [word for word in words if word not in stopwords_def]

    # select NN*
    words_tag = pos_tag(words_flt)
    words_fin = []
    for word_tag in words_tag:
        if word_tag[1].startswith('NN'):  # select NN, NNS, ...
            words_fin.append(word_tag[0])

    return words_fin


def preprocessing(datasets_path, target_name):
    preprocessed_path = os.path.join('./results', target_name + '.pkl')
    # 기전처리된 파일 사용 시
    # if os.path.isfile(preprocessed_path):
    #     with open(preprocessed_path, 'rb') as fin:
    #         df_target = pickle.load(fin)
    #     fin.close()
    #     return df_target

    #데이터 로드
    # if target_name == 'news_new_2':
    #     # FIXME csv로 read할 경우 to_datetime에서 parsing 오류 발생 (message : time data 2008. ")
    #     target_path = os.path.join(datasets_path, target_name + '.xlsx')
    #     df_target = pd.read_excel(target_path)
    # else:
    #target_path = os.path.join(datasets_path, target_name + '.csv') #.csv
    #df_target = pd.read_csv(target_path)
    df_target = data(datasets_path, target_name)

    df_target['date'] = pd.to_datetime(df_target['date'], format='%Y-%m-%d')
    text = df_target['text'].astype(str).values.tolist()

    # 전처리
    documents = []
    stopwords_path = os.path.join('Stopword_Eng_Blockchain.txt')
    for document in text:
        words = _preprocess_text(document, stopwords_path)
        #if len(words) > 0:
        documents.append(words)
    df_target['text'] = documents

    # 전처리된 결과 저장
    with open(preprocessed_path, 'wb') as fout:
        pickle.dump(df_target, fout)
    fout.close()

    return df_target


if __name__ == '__main__':

    target_name = 'news_new'
    datasets_path = os.path.abspath('./data')
    print(datasets_path)

    _data = data(datasets_path, target_name)
    print(_data)

    df_target = preprocessing(datasets_path, target_name)
    print(df_target)


# # 어떻게 생겼나 확인
# df1 = pd.read_pickle('./results/news_new.pkl')
# #print(df1)
# print(df1.head(40))
