import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
my_token = os.getenv("NEWSDATA")
url = "https://newsdata.io/api/1/news?apikey=" + my_token + "&language=" + "en" + "&timeframe=" + "24" + "&page=1703173688663966910"
response = requests.get(url)
data = response.json()

try:
    response.raise_for_status()
    print("Request was successful!")
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
pass