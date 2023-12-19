from sklearn.datasets import fetch_20newsgroups

# Get labeled data
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data['data']
y = data['target']


from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

# Get labeled data
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data['data']
y = data['target']

# Skip over dimensionality reduction, replace cluster model with classifier,
# and reduce frequent words while we are at it.
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)



from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from transformers import AutoTokenizer, AutoModel

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts = dataset["abstract"]
titles = dataset["title"]

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
encoded_input = tokenizer(abstracts[0], padding=True, truncation=True, return_tensors='pt')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentences = [sent_tokenize(abstract) for abstract in abstracts]
paragraphs = []
for doc in sentences:
    length = 0
    paragraph = ""
    for sentence in doc:
        length += len(tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').encodings[0].tokens)
        paragraph += sentence
        if length > 256:
            break
    paragraphs.append(paragraph)
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
  umap_model=empty_dimensionality_model,
  hdbscan_model=clf,
  ctfidf_model=ctfidf_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

y = ["uncertainty estimation, AUtoencoders, Time series, Deep Networks"]

topics, probs = topic_model.fit_transform(abstracts, embeddings, y = y)
info = topic_model.get_topic_info()

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("Models/bertopic_pickle", serialization="pickle", save_ctfidf=True, save_embedding_model=embedding_model)
topic_model.save("Models/bertopic_safetensor", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data['data']

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
pass