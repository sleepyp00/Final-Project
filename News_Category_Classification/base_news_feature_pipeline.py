import numpy as np 
import pandas as pd
import hopsworks
import torch
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
from utils import train_ensemble_standard, train_model, accuracy, get_dataloaders
import joblib
import pandas as pd

"""
Prepares and uploads the news category dataset to Hopsworks
"""

df = pd.read_json('Data/News_Category_Dataset_v3.json', lines = True)
df = df.drop_duplicates(subset='headline')
df = df.drop(columns=['link', 'headline', 'authors', 'date'])
unique_values = df['category'].unique()

# Convert the unique values to a list
unique_values_list = list(unique_values)

category_to_index = {
    "POLITICS":0,
    "TECH":1,
    "SCIENCE":1,
    "ENTERTAINMENT":2,
    "SPORTS":3,
    "BUSINESS":4,
    "MONEY":4,
}

index_to_category = {
    0:"politics",
    1:"science",
    2:"entertainment",
    3:"sports",
    4:"business"
}

value_counts = df['category'].value_counts()
summaries = df['short_description'].values
categories = df['category'].values
targets = np.zeros(len(categories), dtype=int)
for i, value in enumerate(categories):
    if value in category_to_index:
        targets[i] = category_to_index[value]
    else:
        targets[i] = -1

unique, counts = np.unique(targets, return_counts=True)

summaries = summaries[targets != -1]
categories = categories[targets != -1]
targets = targets[targets != -1]

#translate categories
categories = []
for target in targets:
    categories.append(index_to_category[target])
categories = np.array(categories)

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()
embedding_model = mr.get_model("news_embedding", version = 1)
model_dir = embedding_model.download()
embedding_model = joblib.load(model_dir + "/news_embedding.pkl")

features = embedding_model.encode(summaries.tolist(), show_progress_bar=True)


data = {"categoryidx":targets.tolist(), 
        "category":categories.tolist()}

data["embedding"] = features.tolist()



try:
    news_fg = fs.get_feature_group(
            name="basedataset",
            version=1,)
    news_fg.delete()
except:
    print("not able to delete old")


def insert_df(dataframe):
    columns = dataframe.columns.tolist()
    news_fg = fs.get_or_create_feature_group(
        name="basedataset",
        version=1,
        primary_key=columns,
        description="Base news category dataset")
    news_fg.insert(dataframe, 
                   write_options = {"wait_for_job":True})

def upload_df_in_splits(dataframe, n_splits:int):
    for data_frame in np.array_split(dataframe, n_splits, axis = 0):
        insert_df(data_frame)


upload_df_in_splits(pd.DataFrame(data), n_splits=10)

