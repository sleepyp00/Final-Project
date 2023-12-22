
import json
from newspaper import Article

import pandas as pd

from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import hopsworks
#import datetime
from datetime import date, timedelta, datetime

project = hopsworks.login()
fs = project.get_feature_store()

""" start_date = (datetime.datetime.now() - datetime.timedelta(days=7))
end_date = (datetime.datetime.now()) """

news_fg = fs.get_feature_group(name="news", version=1)
query = news_fg.select_all()

feature_view = fs.get_or_create_feature_view(name="news",
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

results = {"title":[], "link":[], "content":[]}
for i in range(3):
    for j in range(9):
        with open("object"+str(i) + str(j)+".json", 'r') as file:
        # Use json.load() to deserialize and load the object from the file
            response = json.load(file)
            result = response['results']
            for article in response['results']:
                """ try:
                    a = Article(article['link'], language='en')  
                    a.download()
                    a.parse()
                    text = a.text
                except:
                    text = ""

                results.append({'id':article['article_id'], 
                                'title':article['title'],
                                'link':article['link'],
                                'content':article['content'] if len(article['content']) >= len(text) else text}) """
                
                """ results.append({'id':article['article_id'], 
                                'title':article['title'],
                                'link':article['link'],
                                'content':article['content']}) """
                
                results['title'].append(article['title'])
                results['link'].append(article['link'])
                results['content'].append(article['content'])
            
            
        
        pass
#print(results[0]['content'])
""" from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
docs = []
for result in results:
    docs.append(result['content']) """
#article = results[0]['content']
    
import time
import numpy as np
start_time = time.time()


embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedding_model.encode(results['content'], show_progress_bar=True)
results['embedding'] = embeddings.tolist()

#times = np.full(len(embeddings), fill_value=datetime.datetime.now())
#timestamp = datetime.timestamp(datetime.datetime.now())
results['time'] = date.today()
df = pd.DataFrame(results)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")



columns = df.columns.tolist()

news_fg = fs.get_or_create_feature_group(
    name="news",
    version=1,
    primary_key=columns,
    description="Current News dataset",
    event_time="time")
news_fg.insert(df)


pass