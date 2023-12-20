import numpy as np

import os
from pygooglenews import GoogleNews
import requests
import newsdataapi
from dotenv import load_dotenv
from newspaper import Article
import time

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

class NEWSDATAFeed(NewsFeed):
    def __init__(self, language: str = 'en') -> None:
        super().__init__(language)
        self.api_key = os.getenv("NEWSDATA")
        self.api = newsdataapi.NewsDataApiClient(apikey=self.api_key)
        self.nextPage = None
        
    def get_daily_news(self):
        #limited to 30 credits
        results = []
        for i in range(30):
            response = self.api.news_api(timeframe=24, language=self.language, page = self.nextPage, full_content=True)
            for article in response['results']:
                results.append({'id':article['article_id'], 
                                'title':article['title'],
                                'link':article['link'],
                                'content':article['content']})
            time.sleep(1)
        return results
    
if __name__ == "__main__":
    pass
        

    
