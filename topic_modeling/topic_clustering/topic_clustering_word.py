# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/blockchain/papers_keywords.csv", encoding='utf-8') # news_keywords.csv / papers_keywords.csv / patents_keywords.csv
#data = df['patent_keywords'] #'patent_keywords'
print(df)


# dmr = df[['topic number','patent_keywords']].iloc[0:12]
# print(dmr)
# bert = df[['topic number','patent_keywords']].iloc[12:]
# print(bert)


# news / papers / patents 선택해서 주석제외하고 돌리면 됨 
# news
# dmr = df['dmr_news_keywords'][0:12]
# print(dmr)
# lda = df['lda_news_keywords'][0:12]
# print(lda)

# papers
dmr = df['dmr_papers_keywords'][0:8]
print(dmr)
lda = df['lda_papers_keywords'][0:8]
print(lda)

# # patents
# dmr = df['dmr_patents_keywords'][0:14]
# #print(dmr)
# lda = df['lda_patents_keywords'][0:14]
#print(lda)


sentence1 = dmr.to_list()
#print(sentence1[0].split(" ")[0])
#print(sentence1)
sentence2 = lda.to_list()
#print(sentence2)


corpus = sentence1 + sentence2
corpus

#[['identity', 'authentication', 'access', 'entity', 'record', 'verification', 'server', 'authorization', 'identifier', 'trust'], ['hash', 'file', 'blocks', 'ledger', 'media', 'transactions', 'structure', 'configuration', 'party', 'game'],.....]
cor = []
i=0
for i in range(len(corpus)):
  c = corpus[i].split(" ")
  cor.append(c)
print(cor)



#vocab 
vocab = list(set(sum(cor , [])))
_ = vocab.sort()
print(vocab)
print(len(vocab))

# one-hot encoding 
# one_hot = [[1 if keyword in sequence else 0 for keyword in vocab] for sequence in cor]
# print(one_hot)

terms_to_ids = {}

i=0
for v in vocab:
  terms_to_ids[v] = i
  i+=1
terms_to_ids

#terms_to_ids['access']

## corpus_one_hot
allones = []
for i in range(len(corpus)):
  c = corpus[i]

  #terms_to_ids 
  one_hot = [0]*len(vocab)
  #print(one_hot)
  for x in c.split(" "):
    #print(x)
    one_hot[terms_to_ids[x]] = 1
  print(one_hot)
  allones.append(one_hot)


## dmr_one_hot
allones_dmr = []
for i in range(len(sentence1)):
  c = sentence1[i]

  #terms_to_ids 
  one_hot = [0]*len(vocab)
  #print(one_hot)
  for x in c.split(" "):
    #print(x)
    one_hot[terms_to_ids[x]] = 1
  print(one_hot)
  allones_dmr.append(one_hot)


## lda_one_hot
allones_lda = []
for i in range(len(sentence2)):
  c = sentence2[i]

  #terms_to_ids 
  one_hot = [0]*len(vocab)
  #print(one_hot)
  for x in c.split(" "):
    #print(x)
    one_hot[terms_to_ids[x]] = 1
  print(one_hot)
  allones_lda.append(one_hot)

np.array(allones)
  np.array(allones_dmr)
  np.array(allones_lda)

!pip install transformers

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

import numpy as np

embeddingDict = {}
for i in range(len(vocab)):
  encoding = tokenizer(vocab[i], return_tensors="pt")
  output = model(**encoding)
  embeddings = output[0][0].detach().numpy() #embedding value 뽑히는 것 
  embeddings = embeddings.reshape(embeddings.shape[0], -1)
  avg = np.average(embeddings[1:-1], axis=0)
  embeddingDict[vocab[i]] = avg
#print(embeddingDict.values())

#print(embeddingDict['authentication'])
print(embeddingDict)
#print(embeddingDict.items())

embeddingDict = dict(sorted(embeddingDict.items()))

#embeddingDict.values()

embvalues = np.array(list(embeddingDict.values()))

embvalues.shape

np.dot(np.array(one_hot).reshape(1, -1), embvalues).shape

#allones
finvector = {}
for i in range(len(allones)):
  finvector[i] = np.dot(allones[i], embvalues)

# #allones_dmr
finvector_dmr = {}
for i in range(len(allones_dmr)):
  finvector_dmr[i] = np.dot(allones_dmr[i], embvalues)

# #allones_lda
finvector_lda = {}
for i in range(len(allones_lda)):
  finvector_lda[i] = np.dot(allones_lda[i], embvalues)

finvector[i].shape
# finvector_dmr[i].shape

list(finvector.values())

np.array(finvector).reshape(1, -1)

# import itertools
# from sklearn.metrics.pairwise import cosine_similarity


# for pair in itertools.combinations(list(finvector.keys()), 2):
#   if pair[0] <= 13 and pair[1] >= 14:
#     i=0
#     for i in range(14):
#       pair[0] == i
#       # Find the pairs with the highest cosine similarity scores
#       # pairs = []
#       # for i in range(len(cosine_scores)-1):
#       #     for j in range(i+1, len(cosine_scores)):
#       #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#       print(pair)

#       print(sorted(cosine_similarity(finvector[pair[0]].reshape(1, -1), finvector[pair[1]].reshape(1, -1))))

embeddings = list(finvector.values()) 
embeddings = embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings1 = list(finvector_dmr.values()) 
embeddings1 = embeddings1 /  np.linalg.norm(embeddings1, axis=1, keepdims=True)

embeddings2 = list(finvector_lda.values()) 
embeddings2 = embeddings2 /  np.linalg.norm(embeddings2, axis=1, keepdims=True)

# cosine_scores = cosine_similarity(embeddings1, embeddings2)
# print(cosine_scores)

# # Highst similarity
# # Find the pairs with the highest cosine similarity scores
# pairs = []
# for i in range(len(cosine_scores)-1):
#     for j in range(i+1, len(cosine_scores)):
#         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
# pairs

# #cosine_similarity
# import itertools
# from sklearn.metrics.pairwise import cosine_similarity

# for pair in itertools.combinations(list(finvector.keys()), 2):
#   print(pair)
#   print(cosine_similarity(finvector[pair[0]].reshape(1, -1), finvector[pair[1]].reshape(1, -1)))

print(finvector)

!pip install sentence_transformers

# https://stackoverflow.com/questions/55619176/how-to-cluster-similar-sentences-using-bert #참고 링크

"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
"""
#from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Normalize the embeddings to unit length
embeddings = list(finvector.values()) 
embeddings = embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)
print(embeddings)

# Perform clustering
# n_clusters 랑 distance_threshold 둘 중 하나만 쓸 수 있음 
# n_clusters = 11
clustering_model = AgglomerativeClustering(n_clusters= None, affinity='cosine', linkage='average', distance_threshold=0.07) # distance_threshold=0.4, distance_threshold=1.5 등 #news: distance_threshold=0.07 #papers:distance_threshold=0.07 #papents:distance_threshold=0.07
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])
#print(clustered_sentences)

for i, cluster in clustered_sentences.items():
    print("Cluster ", i+1)
    print(cluster)
    print("")

# Agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

# define the model
#model = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat = clustering_model.fit(embeddings)
yhat_2 = clustering_model.fit_predict(embeddings)
# retrieve unique clusters
clusters = unique(yhat)
# Calculate cluster validation metrics
score_AGclustering_s = silhouette_score(embeddings, yhat.labels_, metric='cosine')
score_AGclustering_c = calinski_harabasz_score(embeddings, yhat.labels_)
score_AGclustering_d = davies_bouldin_score(embeddings, yhat_2)
print('Silhouette Score: %.4f' % score_AGclustering_s)
print('Calinski Harabasz Score: %.4f' % score_AGclustering_c)
print('Davies Bouldin Score: %.4f' % score_AGclustering_d)



# from scipy import sparse
# import scipy
# sparse_matrix= scipy.sparse.csr_matrix(embeddings)

# from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import cosine

# distance_matrix= pairwise_distances(sparse_matrix, metric="cosine")
# print(distance_matrix)

# #cosine score dimension / mean / standardization plot?

# topic_num, keywords -> dataframe과 pickle형태로 저장하기

topic_cluster={}
topic_num = []
clustering = []
for i, cluster in clustered_sentences.items(): #items()쓰면 key와 value 쌍을 얻을 수 있음 
    #print("Cluster ", i+1)
    #print(cluster)
    str = " ".join(set(cluster)) # keywords간의 중복제거 하기 싫으면 set 없애면 됨 
    #print(str)
    clustering.append(str)
    num = i+1 #f"topic {i+1}" 
    topic_num.append(num)
    topic_cluster[num] = str
topic_keywords = topic_cluster
#print(topic_keywords) #key,value 형태 # ex) {'topic 1': 'cryptocurrencies crypto  services cryptocurrencies price', 'topic 2': 'government trade countries china economy growth sector industry country innovation nfts art game auction artists tokens games sale metaverse nft', 'topic 3': 'university health research students school education work program city science insider trends industry reports media book coverage report intelligence insiders', 'topic 4': 'security sanctions department government money hackers law court states case capital startups startup venture investment fund funding ventures fintech firm', 'topic 5': 'facebook law regulators tax government policy rules libra mr president university students school event research program director education science city', 'topic 6': 'nfts art work game media money something york thats things health supply group products logistics food industry sales customers vehicles', 'topic 7': 'services startups capital startup tech venture investment software platform firm law security court money department hackers enforcement case transactions fraud', 'topic 8': 'application transaction network images tables charts states office abstract device stock stocks shares growth cent investment markets price quarter index', 'topic 9': 'banks payments payment services banking currency money transactions credit customers mr things work money something thats lot media dont york', 'topic 10': 'energy food power industry mining supply oil gas electricity carbon crypto assets securities exchange asset trading regulators cryptocurrencies industry exchanges'}
#print(topic_keywords.keys())
#print(topic_keywords[1])
topic_num = topic_num
keywords = clustering
#print(keywords)


# dataframe 형태로 저장 
df1 = {'topic_num' : topic_num, 'keywords' : keywords}
df2 = pd.DataFrame(df1)
print(df2)
df2.to_csv('/content/drive/MyDrive/Colab Notebooks/blockchain/papers_topic_clustering.csv', index=False)


# picKle 형태로 저장 
import pickle
with open('/content/drive/MyDrive/Colab Notebooks/blockchain/papers_topic_clustering.pkl', "wb") as file:
    pickle.dump(topic_keywords, file)

df = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/blockchain/papers_topic_clustering.pkl') #어떻게 생겼나 확인
print(df)

# df = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/blockchain/results/news_topic_clustering.pkl') #어떻게 생겼나 확인
# print(df)

# dendrogram 그리기 
# import libraries
from scipy.cluster.hierarchy import dendrogram

# call fit method with array of sample coordinates passed as a parameter
trained_model = clustering_model.fit(embeddings)

# A method for generating dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# plot dendrogram to visualize clusters
plot_dendrogram(trained_model)

# !pip install sentence_transformers

# from sentence_transformers import SentenceTransformer, util
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # Two lists of sentences
# # sentences1 = ['The cat sits outside']
# # sentences2 = ['The dog plays in the garden']
# #Compute embedding for both lists
# # embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# # embeddings2 = model.encode(sentences2, convert_to_tensor=True)
# #Compute cosine-similarities
# cosine_scores = util.cos_sim(embeddings1, embeddings2)
# #Output the pairs with their score

# # for i in range(len(sentence1)):
# #   for j in range(len(sentence2)):
# #     # if cosine_scores[i][j] > 0.91:
# #     print("{} \t\t {} \t\t  {:.4f}".format(sentence1[i], sentence2[j], cosine_scores[i][j]))


# for j in range(len(sentence2)):
#   for i in range(len(sentence1)):
#     if cosine_scores[i][j] > 0.9:
#       print("{} \t\t {} \t\t  {:.4f}".format(sentence2[j], sentence1[i], cosine_scores[i][j]))


















