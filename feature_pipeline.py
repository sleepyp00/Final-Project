import os
import requests
from newspaper import Article
import time
import hopsworks
import pickle
import modal
from modal import Stub, Volume, Image
from pathlib import Path

from sentence_transformers import SentenceTransformer
from datetime import date, timedelta, datetime
import pandas as pd



class NewsFeed:
    def __init__(self, language:str = 'en') -> None:
        self.language = language

    def get_daily_news(self):
        pass

    def load_article(self, url:str):
        a = Article(url, language=self.language)
        a.download()
        a.parse()
        return a.text()
    
    def save_state():
        return False
    
    def on_load(self):
        pass

class NEWSDATAFeed(NewsFeed):
    def __init__(self, language: str = 'en', timeframe:str = "24", start_page:str = None, today:date = date.today()) -> None:
        super().__init__(language)
        self.nextPage = start_page
        self.base_url = self.prepare_base_url(language, os.environ["NEWSDATA"], timeframe)
        self.today = today
        self.timeframe = timeframe

        
    def get_daily_news(self):
        #limited to 30 credits at a time, we use 20 to have some margin
        results = {"title":[], "link":[], "content":[]}
        for i in range(20):
            response = requests.get(self.get_next_page())
            try:
                response.raise_for_status()
                data = response.json()
                self.nextPage = data['nextPage']
                for article in data['results']:
                    if self.is_valid_article(article):
                        results['title'].append(article['title'])
                        results['link'].append(article['link'])
                        results['content'].append(article['content'])
                time.sleep(1)
            except requests.exceptions.HTTPError as err:
                self.nextPage = None
                print(f"HTTP error occurred: {err}")
                break
        return results
    
    def is_valid_article(self, article):
        is_valid = article['title'] is not None
        is_valid = is_valid and article['link'] is not None
        is_valid = is_valid and article['content'] is not None
        return is_valid
    
    def prepare_base_url(self, language:str, api_key:str, timeframe:str):
        return "https://newsdata.io/api/1/news?apikey=" + api_key + "&language=" + language + "&timeframe=" + timeframe
            
        
    def get_next_page(self):
        if self.nextPage is not None:
            return self.base_url + "&page=" + self.nextPage
        else:
            return self.base_url
    
    def on_load(self):
        self.base_url = self.prepare_base_url(self.language, os.environ["NEWSDATA"], self.timeframe)
        if self.today != date.today():
            self.today = date.today()
            self.nextPage = None
        
def load_feed(dataset_api):
    try:
        if dataset_api.exists("Resources/FinalProject/NEWSDATAFeed.pkl"):
            dataset_api.download("Resources/FinalProject/NEWSDATAFeed.pkl", overwrite=True)
            with open("NEWSDATAFeed.pkl", "rb") as file:
                news_feed = pickle.load(file)
        else:
            news_feed = NEWSDATAFeed()
    except:
        news_feed = NEWSDATAFeed()
    news_feed.on_load()

    return news_feed

def store_news_features(project, results):
    fs = project.get_feature_store()

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = embedding_model.encode(results['content'], show_progress_bar=True)
    results['embedding'] = embeddings.tolist()
    results['time'] = date.today()
    df = pd.DataFrame(results)

    columns = df.columns.tolist()

    news_fg = fs.get_or_create_feature_group(
        name="news",
        version=1,
        primary_key=columns,
        description="Current News dataset",
        event_time="time")
    news_fg.insert(df)

def f():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    news_feed = load_feed(dataset_api)

    results = news_feed.get_daily_news()

    with open("NEWSDATAFeed.pkl", "wb") as file:
        pickle.dump(news_feed, file)

    dataset_api.upload("NEWSDATAFeed.pkl", "Resources/FinalProject", overwrite=True)

    if len(results['link']) > 0:
        store_news_features(project, results)



stub = Stub(name = "news_daily")
image = Image.debian_slim(python_version="3.10").pip_install(["hopsworks",
                                        "requests",
                                        "newspaper3k",
                                        "sentence-transformers",
                                        "pandas"]) 

@stub.function(image=image, schedule=modal.Period(hours = 2), secrets=[modal.Secret.from_name("HOPSWORKS_API_KEY"),
                                    modal.Secret.from_name("NEWSDATA")])
def g():
    f()

    
if __name__ == "__main__":
    with stub.run():
        g.remote()
    





        








        

    
