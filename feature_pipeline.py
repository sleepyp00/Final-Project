import numpy as np

import os
from pygooglenews import GoogleNews
import requests
import newsdataapi
from dotenv import load_dotenv
from newspaper import Article
import time
import hopsworks
import pickle
import tempfile

# Load environment variables from .env file
load_dotenv()

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

class NEWSDATAFeed(NewsFeed):
    def __init__(self, language: str = 'en', timeframe:str = "24", start_page:str = None) -> None:
        super().__init__(language)
        self.nextPage = start_page
        self.base_url = self.prepare_base_url(language, os.getenv("NEWSDATA"), timeframe)
        
    def get_daily_news(self):
        #limited to 30 credits at a time
        results = {"title":[], "link":[], "content":[]}
        for i in range(1):
            response = requests.get(self.get_next_page())
            try:
                response.raise_for_status()
                data = response.json()
                self.nextPage = data['nextPage']
                for article in data['results']:
                    results['title'].append(article['title'])
                    results['link'].append(article['link'])
                    results['content'].append(article['content'])
                time.sleep(1)
            except requests.exceptions.HTTPError as err:
                self.nextPage = None
                print(f"HTTP error occurred: {err}")
                break
        return results
    
    def prepare_base_url(self, language:str, api_key:str, timeframe:str):
        return "https://newsdata.io/api/1/news?apikey=" + api_key + "&language=" + language + "&timeframe=" + timeframe
            
        
    def get_next_page(self):
        if self.nextPage is not None:
            return self.base_url + "&page=" + self.nextPage
        else:
            return self.base_url
        
    def save_state(self):
        return self.nextPage is not None
        
    
if __name__ == "__main__":
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    try:
        if dataset_api.exists("Resources/FinalProject/NEWSDATAFeed.pkl"):
            dataset_api.download("Resources/FinalProject/NEWSDATAFeed.pkl", overwrite=True)
            with open("NEWSDATAFeed.pkl", "rb") as file:
                newsdata_feed = pickle.load(file)
        else:
            newsdata_feed = NEWSDATAFeed()
    except:
        newsdata_feed = NEWSDATAFeed()


    results = newsdata_feed.get_daily_news()
    if newsdata_feed.save_state():
        with open("NEWSDATAFeed.pkl", "wb") as file:
            pickle.dump(newsdata_feed, file)
            
        dataset_api.upload("NEWSDATAFeed.pkl", "Resources/FinalProject", overwrite=True)




        

    
