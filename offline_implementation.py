import numpy as np

import os
from pygooglenews import GoogleNews
import requests
import newsdataapi


gn = GoogleNews()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the token
""" my_token = os.getenv("NEWSDATA")
api = newsdataapi.NewsDataApiClient(apikey=my_token)
response = api.news_api(timeframe=24, language="en") """

#top = gn.top_news()
#print(top)

#get text of a new article
from newspaper import Article
a = Article("https://www.nation.lk/online/sen-warren-targets-former-officials-amid-crypto-legislation-accusations-246668.html", language='en')
a.download()
a.parse()
print(a.text)
pass