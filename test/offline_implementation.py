import numpy as np
import json
import os
from pygooglenews import GoogleNews
import requests
import newsdataapi
import time


gn = GoogleNews()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the token
my_token = os.getenv("NEWSDATA")
api = newsdataapi.NewsDataApiClient(apikey=my_token)
response = api.news_api(timeframe=24, language="en", scroll=True, max_result=5)
with open("object.json", 'w') as file:
    json.dump(response, file)

with open("object.json", 'r') as file:
# Use json.load() to deserialize and load the object from the file
    response = json.load(file)

for i in range(5):
    for j in range(10):
        response = api.news_api(timeframe=24, language="en", page=response['nextPage'], full_content=True)
        with open("object" +str(i) + str(j)+ ".json", 'w') as file:
            json.dump(response, file)
    
    time.sleep(2)

response = api.news_api(timeframe=24, language="en", page=response['nextPage'])

#top = gn.top_news()
#print(top)

#get text of a new article
from newspaper import Article
a = Article("https://www.nation.lk/online/sen-warren-targets-former-officials-amid-crypto-legislation-accusations-246668.html", language='en')
a.download()
a.parse()
print(a.text)
pass