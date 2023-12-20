from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np

from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
#print("Max Sequence Length:", embedding_model.max_seq_length)

# Change the length to 200
#embedding_model.max_seq_length = 200

#print("Max Sequence Length:", embedding_model.max_seq_length)

df = pd.read_json('my_dataframe.json', orient='records')
abstracts = df['documents'].values
embeddings = np.array(df['embeddings'].values.tolist())
from umap import UMAP

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# KeyBERT
keybert_model = KeyBERTInspired()


#

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
}

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)


#docs = [str(i) for i in range(len(abstracts))]

topics, probs = topic_model.fit_transform(abstracts, embeddings)
info = topic_model.get_topic_info()

embedding_model = "sentence-transformers/all-mpnet-base-v2"
topic_model.save("Models/bertopic_pickle", serialization="pickle", save_ctfidf=True, save_embedding_model=embedding_model)
topic_model.save("Models/bertopic_safetensor", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

pass