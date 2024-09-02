import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set file paths
paper_path = "./results/paper_topic_clustering.csv"
patent_path = "./results/patent_topic_clustering.csv"
news_path = "./results/news_topic_clustering.csv"

# Load files
df_paper = pd.read_csv(paper_path)
df_patent = pd.read_csv(patent_path)
df_news = pd.read_csv(news_path)

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/stsb-roberta-large')

# Generate text embeddings
paper_embeddings = model.encode(df_paper['keywords'].tolist(), convert_to_tensor=True)
patent_embeddings = model.encode(df_patent['keywords'].tolist(), convert_to_tensor=True)
news_embeddings = model.encode(df_news['keywords'].tolist(), convert_to_tensor=True)

# Move embeddings from GPU to CPU and convert to numpy arrays
paper_embeddings = paper_embeddings.cpu().numpy()
patent_embeddings = patent_embeddings.cpu().numpy()
news_embeddings = news_embeddings.cpu().numpy()

# Find similar topics across all three sources
def find_triplet_similar_topics(paper_embeddings, patent_embeddings, news_embeddings, df_paper, df_patent, df_news, threshold=0.7):
    results = []
    for i, j, k in itertools.product(range(len(paper_embeddings)), range(len(patent_embeddings)), range(len(news_embeddings))):
        
        sim_paper_patent = cosine_similarity([paper_embeddings[i]], [patent_embeddings[j]])[0][0]
        sim_paper_news = cosine_similarity([paper_embeddings[i]], [news_embeddings[k]])[0][0]
        sim_patent_news = cosine_similarity([patent_embeddings[j]], [news_embeddings[k]])[0][0]
        
        if sim_paper_patent > threshold and sim_paper_news > threshold and sim_patent_news > threshold:
            results.append((
                df_paper.iloc[i]['topic_num'], df_patent.iloc[j]['topic_num'], df_news.iloc[k]['topic_num'], 
                sim_paper_patent, sim_paper_news, sim_patent_news,
                df_paper.iloc[i]['keywords'], df_patent.iloc[j]['keywords'], df_news.iloc[k]['keywords']
            ))
            print(f"Paper Topic {df_paper.iloc[i]['topic_num']} is similar to Patent Topic {df_patent.iloc[j]['topic_num']} and News Topic {df_news.iloc[k]['topic_num']}")
            print(f"Sim (Paper-Patent): {sim_paper_patent:.2f}, Sim (Paper-News): {sim_paper_news:.2f}, Sim (Patent-News): {sim_patent_news:.2f}")
    
    return results

# Find similar topics between two sources
def find_similar_topics(embeddings1, embeddings2, df1, df2, threshold=0.7):
    similarities = cosine_similarity(embeddings1, embeddings2)
    results = []
    for i, j in itertools.product(range(similarities.shape[0]), range(similarities.shape[1])):
        if similarities[i, j] > threshold:
            result = (
                df1.iloc[i]['topic_num'], df2.iloc[j]['topic_num'], similarities[i, j],
                df1.iloc[i]['keywords'], df2.iloc[j]['keywords']
            )
            results.append(result)
            print(f"Topic {df1.iloc[i]['topic_num']} is similar to Topic {df2.iloc[j]['topic_num']} with similarity {similarities[i, j]:.2f}")
    return results

# Function to save results to CSV
def save_results_to_csv(results, columns, save_path):
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# Find similar topics across all three sources
triplet_similar_topics = find_triplet_similar_topics(paper_embeddings, patent_embeddings, news_embeddings, df_paper, df_patent, df_news, threshold=0.7)
save_results_to_csv(triplet_similar_topics, [
    'Paper Topic Number', 'Patent Topic Number', 'News Topic Number', 
    'Similarity (Paper-Patent)', 'Similarity (Paper-News)', 'Similarity (Patent-News)',
    'Paper Keywords', 'Patent Keywords', 'News Keywords'
], './triplet_similar_topics.csv')

# Find similar topics between two sources
paper_patent_similarities = find_similar_topics(paper_embeddings, patent_embeddings, df_paper, df_patent, threshold=0.7)
save_results_to_csv(paper_patent_similarities, [
    'Paper Topic Number', 'Patent Topic Number', 'Similarity', 
    'Paper Keywords', 'Patent Keywords'
], './paper_patent_similar_topics.csv')

paper_news_similarities = find_similar_topics(paper_embeddings, news_embeddings, df_paper, df_news, threshold=0.7)
save_results_to_csv(paper_news_similarities, [
    'Paper Topic Number', 'News Topic Number', 'Similarity', 
    'Paper Keywords', 'News Keywords'
], './paper_news_similar_topics.csv')

patent_news_similarities = find_similar_topics(patent_embeddings, news_embeddings, df_patent, df_news, threshold=0.7)
save_results_to_csv(patent_news_similarities, [
    'Patent Topic Number', 'News Topic Number', 'Similarity', 
    'Patent Keywords', 'News Keywords'
], './patent_news_similar_topics.csv')
