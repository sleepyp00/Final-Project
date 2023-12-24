from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np

from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import hopsworks
import joblib
from datetime import date, timedelta, datetime

project = hopsworks.login()
fs = project.get_feature_store()

news_fg = fs.get_feature_group(name="news", version=1)
query = news_fg.select_all()

feature_view = fs.get_or_create_feature_view(name="news_view",
                                  version=1,
                                  description="Read from news dataset",
                                  query=query)

today = datetime.now().date()
start_date = today - timedelta(days=7)
tomorrow = today + timedelta(days=1)

df = feature_view.get_batch_data(
        start_time=start_date,
        end_time=tomorrow
    )




mr = project.get_model_registry()

def load_hopsworks_model(name:str, version:int = 1):
    model = mr.get_model(name, version = version)
    model_dir = model.download()
    model = joblib.load(model_dir + "/"+name+".pkl")
    return model
    

""" embedding_model = load_hopsworks_model("news_embedding", version = 1)
umap_model = load_hopsworks_model("news_umap", version = 1)
hdbscan_model = load_hopsworks_model("news_hbdscan", version = 1)
vectorizer_model = load_hopsworks_model("news_vectorizer", version = 1)
representation_model = load_hopsworks_model("news_representation", version = 1) """

from umap import UMAP
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
#umap_model = UMAP()

from hdbscan import HDBSCAN

#hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# KeyBERT
keybert_model = KeyBERTInspired()


from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

But we only want to focus on the kewords that describe the topic:
[KEYWORDS]

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following keywords:
[KEYWORDS]

Based on the information about the topic above, please create a general and 3-5 word label for this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-alpha-GGUF",
    model_file="zephyr-7b-alpha.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50,
    hf=True
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

# Pipeline
generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)

#from bertopic.representation import TextGeneration
from custom_topics import Custom_TextGeneration

# Text generation with Zephyr
#zephyr = Custom_TextGeneration(generator, prompt=prompt, nr_docs=3, doc_length=60, tokenizer="vectorizer")
zephyr = Custom_TextGeneration(generator, prompt=prompt)

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
    "Zephyr": keybert_model
}

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,
  top_n_words=10,
  # Hyperparameters
  verbose=True
)

""" topic_model = BERTopic(

  # Pipeline models
  #embedding_model=embedding_model,
  #umap_model=umap_model,
  #hdbscan_model=hdbscan_model,
  #vectorizer_model=vectorizer_model,
  representation_model=summarizer,

  # Hyperparameters
  verbose=True
) """

documents = df['content'].values
#contents = documents.tolist()
#embed = df['embedding'].values.tolist()
embeddings = np.array(df['embedding'].values.tolist())
#docs = [str(i) for i in range(len(abstracts))]
#documents = documents.tolist()
#emb = embedding_model.encode(documents[3:], show_progress_bar=True)
#emb2 = emb.tolist()
v = documents[documents == None]
docs = documents[documents != None]
embeddings = embeddings[documents != None, :]
v = docs[docs == None]
v2 = embeddings[embeddings == None]
topics, probs = topic_model.fit_transform(docs, embeddings)
topic_model.update_topics(docs, representation_model=keybert_model)
info = topic_model.get_topic_info()
topic_list = info['KeyBERT'].values.tolist()
topic_list1 = info['Zephyr'].values.tolist()
topic_scores = topic_model.get_topics()

titles = df['title'].values
links = df['link'].values

titles = titles[documents != None]
links = links[documents != None]
document_data = {"topic":topics, 
              "probability":probs, 
              "title":titles, 
              "link":links}
document_dataframe = pd.DataFrame(document_data)


topic_dataframe = info
keywords = [None]*len(topic_scores)
scores = [None]*len(topic_scores)
for topic, values in topic_scores.items():
    keyword = [value[0] for value in values]
    score = [value[1] for value in values]

    keywords[topic + 1] = keyword
    scores[topic + 1] = score
topic_dataframe["keywords"] = keywords
topic_dataframe["scores"] = keywords



""" embedding_model = "sentence-transformers/all-mpnet-base-v2"
topic_model.save("Models/bertopic_pickle", serialization="pickle", save_ctfidf=True)
topic_model.save("Models/bertopic_safetensor", serialization="safetensors", save_ctfidf=True)
pass """

#fs = project.get_feature_store()

def save_dataframe(df, name:str, version:int = 1, description:str = "", overwrite:bool = True):
  columns = df.columns.tolist()

  fg = fs.get_or_create_feature_group(
      name=name,
      version=version,
      primary_key=columns,
      description=description)
  fg.insert(df, overwrite = overwrite)

save_dataframe(document_dataframe,
               name = "daily_document_info",
               version = 1,
               description="info about today's news documents")

save_dataframe(topic_dataframe,
               name = "daily_topic_info",
               version = 1,
               description="topic summary of today's news")



