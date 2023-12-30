import os
import requests
from newspaper import Article
import time
import hopsworks
import pickle
import joblib

from sentence_transformers import SentenceTransformer
from datetime import date, timedelta, datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from pathlib import Path
from utils import replace_model

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True)


vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# KeyBERT
keybert_model = KeyBERTInspired()

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
}


    



project = hopsworks.login()
mr = project.get_model_registry()

replace_model(mr,
              embedding_model,
              name="news_embedding",
              version=1,
              description="Model used to create embeddings from news documents")

replace_model(mr,
              umap_model,
              name="news_umap",
              version=1,
              description="Model used for UMAP")

replace_model(mr,
              hdbscan_model,
              name="news_hbdscan",
              version=1,
              description="Model used for hdbscan")

replace_model(mr,
              vectorizer_model,
              name="news_vectorizer",
              version=1,
              description="Model used for vectorization")

replace_model(mr,
              representation_model,
              name="news_representation",
              version=1,
              description="Model used for topic representation")





