from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))["data"]

seed_topic_list = [["North America"], ["Middle East"], ["Sports"], ["Politics"], ["Weather", "climate"], ["Economics"]]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
topics, probs = topic_model.fit_transform(docs)
similar_topics, similarity = topic_model.find_topics("Middle East", top_n=5)
info = topic_model.get_topic_info()
A = topic_model.get_topic("Middle East")
pass