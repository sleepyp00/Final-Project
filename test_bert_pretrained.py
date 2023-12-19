from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# Define embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load model and add embedding model
#loaded_model = BERTopic.load("Models/bertopic_safetensor", embedding_model=embedding_model)
loaded_model = BERTopic.load("Models/bertopic_pickle", embedding_model=embedding_model)
similar_topics, similarity = loaded_model.find_topics("Deep Learning", top_n=5)
info = loaded_model.get_topic_info()
desired_row = info.loc[info['Topic'] == similar_topics[0]]
pass
