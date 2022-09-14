import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import string



def preprocess(text, stopfile): #text는 string이 와야함 #stopfile : 경로
    text = text.lower()

    text_p = "".join([char for char in text if char not in string.punctuation])

    words = word_tokenize(text_p)

    stop_words = stopwords.words('english') #nltk 기본 stopwords #리스트로 가져옴
    customized_stop = [line.strip() for line in open(stopfile, encoding='utf-8')] #우리 stopwords #리스트로 가져옴
    stop_words.extend(customized_stop)
    filtered_words = [word for word in words if word not in stop_words] #stop_words = customized_stop 으로 바꿔도 됨 (만약 nltk 기본 stopword가 싫으면)

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]

    pos = pos_tag(filtered_words)

    return words, filtered_words, stemmed, pos

#column = ['date'],['text']
df = pd.read_csv('./patents_new_100.csv', dtype=str).fillna("") #papers_new.csv / patents_new.csv / news_new.csv


row = len(df['text'].to_list())

for i in range(row):
    #한 문헌에 대해서 pre-process
    #paeprs/patents/news 에 맞게 주석 바꿔달기
    words, filtered_words, stemmed, pos = preprocess(df['text'].to_list()[i], "./Stopword_Eng_Blockchain.txt")

    #df['title+keywords+abstract'][i] = {"words": words, "filtered_words": filtered_words, "stemmed":stemmed, "pos":pos}

    nn_lst = []
    for j in pos:
        if j[1] == "NN": #튜플에서 value값이 NN인 것만 고르기
            nn_lst.append(j[0]) #튜플에서 key값만 가져와서 append
    df['text'][i] = nn_lst # 리스트 형태로 나옴
    #df['title+keywords+abstract'][i] = {"pos": nn_lst} #딕셔너리 형태 안에 리스트 형태로 나옴


    print(df)


#csv로 저장
#df.to_csv('C:/Users/yejin/PycharmProjects/treform/sample_data/for_paper/pre_patents.csv', index=False)



#pickle로 저장 : 방법1, 방법2 중에 하나로 하면 됨
#방법 1
#df.to_pickle('C:/Users/yejin/PycharmProjects/blockchain/result/pre_patents.pkl')

#방법 2
import pickle
with open("./pre_patents.pkl", "wb") as file:
    pickle.dump(df, file)


#파일 내용 보기
df1 = pd.read_pickle('./pre_patents.pkl')
print(df1['text'].head(5))