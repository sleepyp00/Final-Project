from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from transformers import AutoTokenizer, AutoModel

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts = dataset["abstract"]
titles = dataset["title"]

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
#encoded_input = tokenizer(abstracts[0], padding=True, truncation=True, return_tensors='pt')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

""" sentences = [sent_tokenize(abstract) for abstract in abstracts]
paragraphs = []
for doc in sentences:
    length = 0
    paragraph = ""
    for sentence in doc:
        length += len(tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').encodings[0].tokens)
        paragraph += sentence
        if length > 256:
            break
    paragraphs.append(paragraph) """
    #while length < 256:


#sentences = [sentence for doc in sentences for sentence in doc]

from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#print("Max Sequence Length:", embedding_model.max_seq_length)

# Change the length to 200
#embedding_model.max_seq_length = 200

#print("Max Sequence Length:", embedding_model.max_seq_length)
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

df = pd.DataFrame({'documents':abstracts, 'embeddings':embeddings.tolist()})
df.to_json('my_dataframe.json', orient='records')
from umap import UMAP

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

topics, probs = topic_model.fit_transform(abstracts, embeddings)
info = topic_model.get_topic_info()

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("Models/bertopic_pickle", serialization="pickle", save_ctfidf=True, save_embedding_model=embedding_model)
topic_model.save("Models/bertopic_safetensor", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

pass