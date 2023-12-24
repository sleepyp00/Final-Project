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

prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

But we only want to focus on the kewords that describe the topic:
[KEYWORDS]

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""