import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
import pickle

# Load data
def load_data(target_name):
    df1 = pd.read_csv(f'../results/{target_name}_dmr_topic_keywords.csv', encoding='utf-8')
    df2 = pd.read_csv(f'../results/{target_name}_lda_topic_keywords.csv', encoding='utf-8')
    return df1, df2

# Extract keywords and create corpus
def create_corpus(df1, df2):
    dmr_keywords = df1['keywords'].to_list()
    lda_keywords = df2['keywords'].to_list()
    corpus = dmr_keywords + lda_keywords
    return [str(keywords).split(" ") for keywords in corpus]

# Create embeddings using BERT
def create_embeddings(vocab):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    embeddingDict = {}
    for word in vocab:
        encoding = tokenizer(word, return_tensors="pt")
        output = model(**encoding)
        embeddings = output[0][0].detach().numpy()
        avg = np.average(embeddings[1:-1], axis=0)
        embeddingDict[word] = avg
    return embeddingDict

# Perform clustering
def perform_clustering(finvector):
    embeddings = list(finvector.values())
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.05) #patent: 0.05 / paper: 0.07 / news: 0.07
    clustering_model.fit(embeddings)
    return clustering_model

# Save clustering results
def save_clustering_results(target_name, clustered_sentences):
    sorted_topic_nums = sorted(list(clustered_sentences.keys()))
    topic_num_mapping = {old_num: new_num + 1 for new_num, old_num in enumerate(sorted_topic_nums)}

    df_clustering = pd.DataFrame({
        'topic_num': [topic_num_mapping[topic_num] for topic_num in sorted_topic_nums],
        'keywords': [' '.join(set(clustered_sentences[old_num])) for old_num in sorted_topic_nums]
    })
    df_clustering.to_csv(f'./results/{target_name}_topic_clustering.csv', index=False)

    with open(f'./results/{target_name}_topic_clustering.pkl', "wb") as file:
        pickle.dump(clustered_sentences, file)

# Main function
def main(target_name):
    df1, df2 = load_data(target_name)
    corpus = create_corpus(df1, df2)
    vocab = list(set(sum(corpus, [])))
    vocab.sort()
    embeddingDict = create_embeddings(vocab)
    
    # One-hot encoding and vector calculation
    terms_to_ids = {v: i for i, v in enumerate(vocab)}
    allones = [[1 if vocab[j] in c else 0 for j in range(len(vocab))] for c in corpus]
    embvalues = np.array(list(embeddingDict.values()))
    finvector = {i: np.dot(allones[i], embvalues) for i in range(len(allones))}
    
    clustering_model = perform_clustering(finvector)
    
    # Assign clusters and save
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences.setdefault(cluster_id, []).append(' '.join(corpus[sentence_id]))
        
    save_clustering_results(target_name, clustered_sentences)

    # Save scores
    with open(f'./results/{target_name}_score.json', 'w') as f:
        scores = {
            'Silhouette Score': silhouette_score(list(finvector.values()), clustering_model.labels_, metric='cosine'),
            'Calinski Harabasz Score': calinski_harabasz_score(list(finvector.values()), clustering_model.labels_),
            'Davies Bouldin Score': davies_bouldin_score(list(finvector.values()), clustering_model.labels_)
        }
        json.dump(scores, f, indent=4)

# Execute
if __name__ == '__main__':
    target_name = 'patent'
    main(target_name)
